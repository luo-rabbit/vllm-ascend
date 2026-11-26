# SPDX-License-Identifier: Apache-2.0
import pickle
from contextlib import nullcontext
from inspect import unwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config.scheduler import SchedulerConfig
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import RequestStatus


def make_request(request_id, position=97):
    return SimpleNamespace(
        request_id=request_id,
        status=RequestStatus.RUNNING,
        is_prefill_chunk=False,
        num_prompt_tokens=64,
        max_tokens=512,
        num_computed_tokens=position,
        num_in_flight_tokens=0,
        num_tokens=position + 1,
        num_output_tokens=position + 1 - 64,
    )


def allocate(cache, request, num_tokens=8):
    return cache.allocate_slots(
        request, num_tokens, num_lookahead_tokens=7, delay_cache_blocks=True
    )


@pytest.fixture
def setup_cache():
    from vllm_ascend.core.dspark_next_page import DraftPageReservations

    def setup(blocks=20):
        common = {"num_kv_heads": 1, "head_size": 32}
        target = FullAttentionSpec(block_size=256, dtype=torch.float16, **common)
        draft = SlidingWindowSpec(
            block_size=128, sliding_window=64, dtype=torch.bfloat16, is_draft=True, **common
        )
        config = KVCacheConfig(
            blocks, [], [KVCacheGroupSpec(["target"], target), KVCacheGroupSpec(["draft"], draft)]
        )
        cache = KVCacheManager(config, 1024, 256, 128, enable_caching=True)
        reserves = DraftPageReservations(cache)
        request = make_request("r")
        allocate(cache, request)
        cache.take_new_block_ids()
        return cache, reserves, request

    return setup


def prepare(reserves, request):
    return reserves.prepare([request], {request.request_id: 8}, lookahead=7, threshold=16)


def test_evict_hashed_head_and_publish_on_crossing(setup_cache):
    cache, reserves, request = setup_cache(blocks=4)
    pool = cache.block_pool
    visible = cache.get_block_ids("r")
    head = next(pool.free_block_queue.iter_blocks_after(None))
    key = b"cached-prefix-key"
    pool._insert_block_hash(key, head, 128)
    assert pool.cached_block_hash_to_block.get_one_block(key) is head

    assert prepare(reserves, request) == [head.block_id]
    assert pool.cached_block_hash_to_block.get_one_block(key) is None
    assert head.block_hash is None
    assert pool.get_num_free_blocks() == 0
    assert cache.get_block_ids("r") == visible
    assert cache.take_new_block_ids() == []

    request.num_computed_tokens = 113
    assert allocate(cache, request).get_block_ids() == ([], [])
    request.num_computed_tokens = 114
    assert allocate(cache, request).get_block_ids() == ([], [head.block_id])
    assert allocate(cache, request).get_block_ids() == ([], [])
    assert cache.take_new_block_ids() == []


def test_additional_draft_and_target_pages_still_zero(setup_cache):
    cache, reserves, request = setup_cache()
    (spare,) = prepare(reserves, request)
    target, draft = allocate(cache, request, num_tokens=170).get_block_ids()
    assert len(target) == 1
    assert len(draft) == 2
    assert draft[0] == spare
    assert sorted(cache.take_new_block_ids()) == sorted(target + draft[1:])


def test_pressure_reclaims_then_retries_original_allocation(setup_cache):
    cache, reserves, request = setup_cache(blocks=5)
    (spare,) = prepare(reserves, request)
    assert allocate(cache, make_request("other")) is not None
    assert not reserves.pages
    assert spare in cache.take_new_block_ids()


@pytest.mark.parametrize("deferred", [False, True])
def test_request_release_accounts_for_unpublished_page(setup_cache, deferred):
    cache, reserves, request = setup_cache()
    (spare,) = prepare(reserves, request)
    if deferred:
        free_before = cache.block_pool.get_num_free_blocks()
        held = cache.pop_blocks_for_free(request)
        assert [block.block_id for block in held].count(spare) == 1
        assert cache.block_pool.get_num_free_blocks() == free_before
        cache.block_pool.free_blocks(reversed(held))
    else:
        cache.free(request)
    assert not reserves.pages
    assert cache.block_pool.get_num_free_blocks() == 19


def test_scheduler_output_preserves_fields_and_preparation_through_pickle(setup_cache):
    from vllm_ascend.core.dspark_next_page import DSparkNextPageAsyncScheduler

    _, reserves, request = setup_cache()
    scheduler = object.__new__(DSparkNextPageAsyncScheduler)
    scheduler._draft_pages = reserves
    scheduler.requests = {"r": request}
    scheduler.num_lookahead_tokens = 7
    scheduler.num_spec_tokens = 7
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"r": 8}
    output.new_block_ids_to_zero = [55]
    scheduler._prepare_draft_pages(output)
    prepared = reserves.pages["r"].block_id
    copied = pickle.loads(pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL))
    assert copied.draft_block_ids_to_prepare == [prepared]
    assert copied.new_block_ids_to_zero == [55]


@pytest.mark.parametrize("enabled,async_scheduling", [(False, True), (True, True), (True, False)])
def test_additional_config_selects_the_matching_scheduler(monkeypatch, enabled, async_scheduling):
    from vllm_ascend import platform
    from vllm_ascend.ascend_config import AscendConfig, SparseKVOffloadConfig
    from vllm_ascend.core.dspark_next_page import DSparkNextPageAsyncScheduler, DSparkNextPageScheduler

    monkeypatch.setattr(platform, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: False))
    monkeypatch.setattr(platform, "refresh_block_size", lambda _: None)
    config = SimpleNamespace(
        use_v2_model_runner=True,
        kv_transfer_config=None,
        cache_config=None,
        speculative_config=SimpleNamespace(method="dspark"),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm5_next_text")),
        parallel_config=SimpleNamespace(
            worker_cls="already_selected",
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        scheduler_config=SchedulerConfig(max_model_len=1024, is_encoder_decoder=False, async_scheduling=async_scheduling),
    )
    options = {"dspark_prepare_next_page": True} if enabled else {}
    ascend_config = AscendConfig(sparse_kv_offload_config=SparseKVOffloadConfig(), **options)
    platform._setup_worker_and_scheduler(config, ascend_config)
    if enabled:
        selected = DSparkNextPageAsyncScheduler if async_scheduling else DSparkNextPageScheduler
        assert config.scheduler_config.get_scheduler_cls() is selected
    else:
        assert config.scheduler_config.scheduler_cls is None


@pytest.mark.parametrize(
    ("mode", "ids", "complete", "failure", "graph_zero", "fallback"),
    [
        ("relay", None, True, False, [], None),
        ("relay", [27, 33], True, False, [27, -1], [33]),
        ("updatable", [27], True, False, [27, -1], None),
        ("relay", [27], False, False, [], [27]),
        ("eager", [27], True, False, [], [27]),
        ("relay", [27], True, True, [27, -1], [27]),
    ],
)
def test_worker_prepared_page_zero(monkeypatch, mode, ids, complete, failure, graph_zero, fallback):
    from vllm_ascend.worker.v2 import aclgraph_utils, model_runner

    runner = object.__new__(model_runner.NPUModelRunner)
    runner._draft_prepare_ids = None
    runner.kv_block_zeroer = Mock()
    for name in ("ascend_config", "vllm_config", "model_state", "kvpp", "_start_dump_data", "_maybe_send_draft_kv"):
        setattr(runner, name, Mock())
    runner.attn_groups = []
    runner.model_state.prepare_engram = None
    runner.model_state.device_metadata = None

    manager = object.__new__(aclgraph_utils.ModelAclGraphManager)
    manager.model_runner = runner
    manager.update_stream = Mock()
    manager.vllm_config = runner.vllm_config
    desc = Mock(num_tokens=32)
    graph = Mock()
    plan = Mock()
    plan.is_complete.return_value = complete
    manager.graphs = {desc: graph}
    manager._zero_window_replay = (desc, graph, plan)

    monkeypatch.setattr(aclgraph_utils, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(aclgraph_utils, "_get_graph_update_backend", lambda _: None)
    monkeypatch.setattr(aclgraph_utils, "use_updatable_graph", lambda _: mode == "updatable")
    for name in ("has_kv_transfer_group", "should_skip_allreduce_across_dp_group"):
        monkeypatch.setattr(model_runner, name, lambda *args: False)
    for name in ("_start_profiling_chunk_timing", "_finish_profiling_chunk_timing"):
        monkeypatch.setattr(model_runner, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(model_runner.GPUModelRunner, "update_requests", lambda *args: None)

    def replay(*args):
        if failure:
            raise RuntimeError("replay failed")

    monkeypatch.setattr(manager, "_graph_relay", replay)
    monkeypatch.setattr(manager, "_updatable_graph_replay", replay)

    def forward(self, output, **kwargs):
        self.update_requests(output)
        if mode != "eager":
            manager.run_fullgraph(desc)

    monkeypatch.setattr(model_runner.GPUModelRunner, "execute_model", forward)
    output = SchedulerOutput.make_empty()
    if ids:
        output.draft_block_ids_to_prepare = ids
    execute = unwrap(model_runner.NPUModelRunner.execute_model)
    if failure:
        with pytest.raises(RuntimeError, match="replay failed"):
            execute(runner, output)
    else:
        execute(runner, output)

    assert [call.args[0] for call in plan.block_id.fill_.call_args_list] == graph_zero
    if fallback:
        runner.kv_block_zeroer.zero_block_ids.assert_called_once_with(fallback)
    else:
        runner.kv_block_zeroer.zero_block_ids.assert_not_called()
    assert not runner._draft_prepare_ids
