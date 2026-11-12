# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor import custom_op
from vllm.model_executor.layers import linear

import vllm_ascend.models.glm5next.kda as model_kda
import vllm_ascend.quantization.configs.modelslim_config as modelslim
import vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 as mxfp8
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig, get_quant_type_for_layer


@pytest.mark.parametrize(
    "model_type,prefix",
    [
        ("glm5_next", "language_model.model.layers.0.self_attn"),
        ("glm5_next_text", "language_model.model.layers.0.self_attn"),
        ("glm5_next_mtp", "model.layers.0.self_attn"),
    ],
)
@pytest.mark.parametrize("quant_type", ["W8A8_MXFP8", "FLOAT", "NATIVE_FP8", "OTHER_LAYER", None])
def test_kda_preserves_only_explicit_modelslim_quantization(monkeypatch, model_type, prefix, quant_type):
    description = {
        f"model.language_model.layers.0.self_attn.{name}.weight": quant_type
        for name in ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")
    }
    quant = AscendModelSlimConfig(description)
    if quant_type == "NATIVE_FP8":
        quant = SimpleNamespace()
    elif quant_type is None:
        quant = None
    elif quant_type == "OTHER_LAYER":
        quant = AscendModelSlimConfig({"model.language_model.layers.1.self_attn.q_proj.weight": "W8A8_MXFP8"})
    config = SimpleNamespace(
        quant_config=quant, model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))
    )
    captured = []

    class StopConstruction(Exception):
        pass

    def base_init(self, hf_config, vllm_config, prefix):
        captured.append(vllm_config.quant_config)
        raise StopConstruction

    monkeypatch.setattr(model_kda, "use_glm_recoverssm", lambda _: False)
    monkeypatch.setattr(model_kda.GatedDeltaNetAttention, "__init__", base_init)
    with pytest.raises(StopConstruction):
        model_kda.Glm5NextLinearAttention(SimpleNamespace(), config, prefix)
    assert captured == [quant if quant_type == "W8A8_MXFP8" else None]
    assert config.quant_config is quant


@pytest.mark.parametrize(
    "model_type,prefix",
    [
        ("glm5_next", "language_model.model.layers.0.self_attn"),
        ("glm5_next_text", "language_model.model.layers.0.self_attn"),
        ("glm5_next_mtp", "model.layers.0.self_attn"),
    ],
)
def test_kda_fused_projection_resolves_all_six_shards(model_type, prefix):
    names = ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"]
    quant = AscendModelSlimConfig(
        {f"model.language_model.layers.0.self_attn.{name}.weight": "W8A8_MXFP8" for name in names}
    )
    quant._update_packed_modules_mapping(model_type)
    assert quant.packed_modules_mapping["in_proj_qkvbfg_a"] == names
    assert (
        get_quant_type_for_layer(
            quant.quant_description,
            prefix + ".in_proj_qkvbfg_a",
            quant.packed_modules_mapping,
            prefix_mapper=lambda name: quant.quant_prefix_mapper(model_type, name),
        )
        == "W8A8_MXFP8"
    )


@pytest.mark.parametrize("tp_size,rank", [(1, 0), (2, 0), (2, 1)])
def test_kda_mxfp8_fused_weights_and_scales_load_with_replicated_shards(monkeypatch, tp_size, rank):
    names = ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"]
    quant = AscendModelSlimConfig(
        {f"model.language_model.layers.0.self_attn.{name}.weight": "W8A8_MXFP8" for name in names}
    )
    config = SimpleNamespace(
        quant_config=quant, model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="glm5_next"))
    )
    monkeypatch.setattr(modelslim, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(mxfp8, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(mxfp8, "get_dynamic_mx_quant_scale_alg", lambda _: 0)
    monkeypatch.delitem(custom_op.op_registry_oot, "ColumnParallelLinear", raising=False)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: rank)
    sizes = [256, 256, 256, 128, 128, 128]
    layer = model_kda._Glm5NextMergedColumnParallelLinear(
        128,
        sizes,
        replicated_shard_ids=(4, 5),
        tp_size=tp_size,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=quant,
        prefix="language_model.model.layers.0.self_attn.in_proj_qkvbfg_a",
    )
    assert isinstance(layer.quant_method.quant_method, mxfp8.AscendW8A8MXFP8DynamicLinearMethod)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.dtype == torch.uint8
    expected_weights, expected_scales = [], []
    for sid, size in enumerate(sizes):
        weight = torch.full((size, 128), sid + 1, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.full((size, 4), sid + 1, dtype=torch.uint8)
        layer.weight.weight_loader(layer.weight, weight, sid)
        layer.weight_scale.weight_loader(layer.weight_scale, scale, sid)
        expected_weights.append(weight if sid in (4, 5) else weight.chunk(tp_size)[rank])
        expected_scales.append(scale if sid in (4, 5) else scale.chunk(tp_size)[rank])
    torch.testing.assert_close(layer.weight.float(), torch.cat(expected_weights).float())
    torch.testing.assert_close(layer.weight_scale, torch.cat(expected_scales))
