from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.worker.v2 import attn_utils


def test_glm_draft_bf16_page_does_not_pad_target_or_recurrent_pages(monkeypatch):
    main_bytes = 4480 * 528
    main = AscendMLAAttentionSpec(
        block_size=4480,
        num_kv_heads=1,
        head_size=528,
        dtype=torch.int8,
        model_version="glm5_next",
        cache_sparse_sfa_c8=True,
        indexes_kv_by_block_stride=True,
    )
    draft = AscendMLAAttentionSpec(
        block_size=4480,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
    )
    recurrent = MambaSpec(
        block_size=4480,
        shapes=((1024,),),
        dtypes=(torch.float32,),
        page_size_padded=main_bytes,
    )
    original = {"model.layers.3.attn": main, "draft.layers.0.attn": draft, "model.layers.0.linear_attn": recurrent}
    layers = {
        name: SimpleNamespace(
            get_kv_cache_spec=lambda _, s=spec: s,
            get_attn_backend=lambda: SimpleNamespace(),
            is_draft_layer=name.startswith("draft."),
        )
        for name, spec in original.items()
    }
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="auto"),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        attention_config=SimpleNamespace(indexer_kv_dtype="auto"),
    )
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_: layers)
    monkeypatch.setattr(attn_utils, "enable_sfa_dcp_replicated_indexer", lambda _: False)
    specs = attn_utils.get_kv_cache_spec(cfg)
    assert specs["model.layers.3.attn"].page_size_bytes == main_bytes
    assert specs["model.layers.0.linear_attn"].page_size_bytes == main_bytes
    assert specs["draft.layers.0.attn"].page_size_bytes == 4480 * 1152
