# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark target features must describe the completed mHC residual stream."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch
from vllm.model_executor.models.interfaces import supports_eagle3

from vllm_ascend.models.glm5next import model as glm


def make_model(monkeypatch, *, sequence_parallel=False):
    model = glm.Glm5NextModel.__new__(glm.Glm5NextModel)
    nn.Module.__init__(model)
    model.do_not_compile = True
    model.config = SimpleNamespace(hc_mult=4)
    model.start_layer = 0
    model.end_layer = 3
    model.is_sequence_parallel = sequence_parallel
    model.norm = nn.Identity()
    monkeypatch.setattr(glm, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    return model


class AddLayer(nn.Module):
    def forward(self, positions, hidden_states, residual, post, comb):
        return hidden_states + 1, None, None, None


class ScaleNorm(nn.Module):
    def forward(self, x):
        return x * 10


@pytest.mark.parametrize("wrapped", [False, True])
def test_dspark_aux_interface_reaches_text_decoder(monkeypatch, wrapped):
    model = make_model(monkeypatch)
    text = glm.Glm5NextForCausalLM.__new__(glm.Glm5NextForCausalLM)
    nn.Module.__init__(text)
    text.model = model
    target = text
    if wrapped:
        target = glm.Glm5NextForConditionalGeneration.__new__(glm.Glm5NextForConditionalGeneration)
        nn.Module.__init__(target)
        target.language_model = text
    assert supports_eagle3(target)
    target.set_aux_hidden_state_layers((2, 22, 38, 58, 74))
    assert model.aux_hidden_state_layers == (2, 22, 38, 58, 74)


def test_aux_materializes_deferred_mhc_without_mutating_state(monkeypatch):
    model = make_model(monkeypatch)
    hidden = torch.arange(12).reshape(3, 4).bfloat16()
    residual = torch.arange(48).reshape(3, 4, 4).bfloat16()
    post = torch.full((3, 4, 1), 0.5)
    comb = torch.eye(4).expand(3, -1, -1).contiguous()
    original = residual.clone()

    def native_post(x, r, p, c):
        return mhc_post_torch(x, r, p.unsqueeze(-1), c)

    monkeypatch.setattr(torch.ops._C_ascend, "npu_hc_post", native_post, raising=False)
    expected = mhc_post_torch(hidden, residual, post, comb).mean(dim=1)
    actual = model._aux_hidden_state(hidden, residual, post, comb)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(residual, original)
    assert model._aux_hidden_state(hidden, None, None, None) is hidden


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_capture_boundaries_preserves_token_order_and_discards_sp_padding(monkeypatch, sequence_parallel):
    model = make_model(monkeypatch, sequence_parallel=sequence_parallel)
    model.norm = ScaleNorm()
    model._active_layers = nn.ModuleList([AddLayer() for _ in range(3)])
    model._set_aux_hidden_state_layers((0, 2, 3))
    inputs = torch.arange(12).reshape(3, 4).bfloat16()
    if sequence_parallel:
        # Model a padded SP shard using CPU tensors. Gathering restores token
        # order; the fourth row must never escape in target or auxiliary output.
        monkeypatch.setattr(glm, "pad_sequence_parallel_tokens", lambda x: torch.cat((x, x.new_zeros(1, 4))))
        monkeypatch.setattr(glm, "sp_shard", lambda x: x)
        monkeypatch.setattr(glm, "sp_all_gather", lambda x: x)
    result, aux = model(None, torch.arange(3), None, inputs)
    torch.testing.assert_close(result, (inputs + 3) * 10)
    assert len(aux) == 3
    for actual, offset in zip(aux, (0, 2, 3)):
        torch.testing.assert_close(actual, inputs + offset)


def test_forward_exports_completed_mhc_feature_at_selected_boundary(monkeypatch):
    model = make_model(monkeypatch)

    class DeferredLayer(nn.Module):
        def forward(self, positions, x, residual, post, comb):
            residual = x.unsqueeze(1).expand(-1, 4, -1).contiguous()
            return x + 1, residual, torch.full((x.shape[0], 4, 1), 0.5), torch.eye(4).expand(x.shape[0], -1, -1)

    model._active_layers = nn.ModuleList([DeferredLayer(), AddLayer(), AddLayer()])
    model._set_aux_hidden_state_layers((1,))
    monkeypatch.setattr(
        torch.ops._C_ascend, "npu_hc_post", lambda x, r, p, c: mhc_post_torch(x, r, p.unsqueeze(-1), c), raising=False
    )
    inputs = torch.arange(12).reshape(3, 4).bfloat16()
    _, aux = model(None, torch.arange(3), None, inputs)
    assert len(aux) == 1
    torch.testing.assert_close(aux[0], inputs + (inputs + 1) * 0.5)


def test_without_aux_capture_retains_tensor_output(monkeypatch):
    model = make_model(monkeypatch)
    model._active_layers = nn.ModuleList([AddLayer()])
    inputs = torch.ones(3, 4).bfloat16()
    output = model(None, torch.arange(3), None, inputs)
    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(output, inputs + 1)
