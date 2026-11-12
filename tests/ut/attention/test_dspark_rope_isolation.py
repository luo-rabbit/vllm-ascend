import torch

from vllm_ascend.attention.mla_v1 import AscendMLAMetadataBuilder
from vllm_ascend.ops import rotary_embedding as rotary


def test_dspark_uses_own_rope_with_nope_target_and_stable_buffers(monkeypatch):
    builder = AscendMLAMetadataBuilder.__new__(AscendMLAMetadataBuilder)
    # Same layout as RotaryEmbedding.cos_sin_cache: half cos then half sin.
    builder._draft_rope_table = torch.arange(10 * 64, dtype=torch.float32).view(10, 64)
    builder._draft_rope_cos = torch.empty(8, 1, 1, 64)
    builder._draft_rope_sin = torch.empty_like(builder._draft_rope_cos)
    target = torch.empty(8, 1, 1, 0)
    monkeypatch.setattr(rotary, "_cos_mla", target)
    monkeypatch.setattr(rotary, "_sin_mla", target)
    positions = torch.tensor([1, 3])
    cos, sin = builder._get_cos_and_sin_mla(positions, use_cache=True)
    assert cos.shape == sin.shape == (2, 1, 1, 64)
    assert cos.data_ptr() == builder._draft_rope_cos.data_ptr()
    assert sin.data_ptr() == builder._draft_rope_sin.data_ptr()
    assert torch.equal(cos[:, 0, 0], builder._draft_rope_table[positions, :32].repeat(1, 2))
    assert torch.equal(sin[:, 0, 0], builder._draft_rope_table[positions, 32:].repeat(1, 2))
    new_cos, _ = builder._get_cos_and_sin_mla(torch.tensor([2, 4]), use_cache=True)
    assert new_cos.data_ptr() == cos.data_ptr()
    assert rotary._cos_mla is target


def test_dspark_initialization_uses_layer_geometry_with_target_config():
    from types import SimpleNamespace as NS

    builder = AscendMLAMetadataBuilder.__new__(AscendMLAMetadataBuilder)
    builder.rope_dim = 0  # MRV2 DSpark retains target model_config.
    builder.nope_zero_rope_cache = {}
    table = torch.empty(16, 64)
    impl = NS(qk_rope_head_dim=64, rotary_emb=NS(cos_sin_cache=table))
    spec = NS(method="dspark")
    builder._init_dspark_rope(impl, spec, 8, torch.device("cpu"))
    assert builder.rope_dim == 64
    assert builder.nope_zero_rope_cache is None
    assert builder._draft_rope_table is table
    assert builder._draft_rope_cos.shape == (8, 1, 1, 64)
