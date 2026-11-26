# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterable, Sequence

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.request import Request


class DraftPageReservations:
    """One private spare per request, outside the draft's visible page table."""

    def __init__(self, cache: KVCacheManager):
        managers = [
            manager
            for manager in cache.coordinator.single_type_managers
            if type(manager) is SlidingWindowManager
            and manager.kv_cache_spec.is_draft
            and manager.records_new_block_ids
            and manager.block_pool is cache.block_pool
        ]
        if len(managers) != 1:
            raise ValueError("dspark_prepare_next_page requires one zeroed draft-only SWA group in the shared KV pool")
        self.manager = manager = managers[0]
        self.pool = cache.block_pool
        self.max_model_len = cache.max_model_len
        self.watermark = cache.watermark_blocks
        self.pages: dict[str, KVCacheBlock] = {}

        # Hook just this scheduler's managers, keeping upstream allocation and
        # admission in charge of all mandatory pages.
        self._count_blocks = manager.get_num_blocks_to_allocate
        self._allocate_blocks = manager.allocate_new_blocks
        self._allocate_slots = cache.allocate_slots
        self._free = cache.free
        self._pop_blocks_for_free = cache.pop_blocks_for_free
        manager.get_num_blocks_to_allocate = self.count_blocks
        manager.allocate_new_blocks = self.allocate_blocks
        cache.allocate_slots = self.allocate_slots
        cache.free = self.free
        cache.pop_blocks_for_free = self.pop_blocks_for_free

    def _tail(self, request_id: str) -> int:
        return len(self.manager.req_to_blocks[request_id]) * self.manager.block_size

    def prepare(
        self, requests: Iterable[Request], scheduled: dict[str, int], lookahead: int, threshold: int
    ) -> list[int]:
        ids = []
        for request in requests:
            if self.pool.get_num_free_blocks() <= self.watermark:
                break
            rid = request.request_id
            if rid in self.pages or not request.num_output_tokens:
                continue
            confirmed = request.num_computed_tokens - request.num_in_flight_tokens
            if confirmed < max(request.num_prompt_tokens, request.num_tokens - 1):
                continue
            tail = self._tail(rid)
            demand = request.num_computed_tokens + scheduled[rid] + lookahead
            if not 0 <= tail - demand <= threshold:
                continue
            if tail >= min(self.max_model_len, request.num_prompt_tokens + request.max_tokens):
                continue
            # The normal allocator evicts a cached prefix at the queue head
            # before handing out its physical page.
            (block,) = self.pool.get_new_blocks(1)
            self.pages[rid] = block
            ids.append(block.block_id)
        return ids

    def count_blocks(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        *args,
        apply_admission_cap: bool = False,
        **kwargs,
    ) -> int:
        count = self._count_blocks(
            request_id,
            num_tokens,
            new_computed_blocks,
            *args,
            apply_admission_cap=apply_admission_cap,
            **kwargs,
        )
        if count and not apply_admission_cap and not new_computed_blocks and self._needed(request_id, num_tokens):
            count -= 1
        return count

    def _needed(self, request_id: str, demand: int) -> bool:
        return request_id in self.pages and demand > self._tail(request_id)

    def allocate_blocks(self, request_id: str, num_tokens: int, num_tokens_main_model: int) -> list[KVCacheBlock]:
        prepared = self.pages.pop(request_id) if self._needed(request_id, num_tokens) else None
        if prepared is not None:
            self.manager.req_to_blocks[request_id].append(prepared)
        new = self._allocate_blocks(request_id, num_tokens, num_tokens_main_model)
        if prepared is not None:
            new.insert(0, prepared)
        return new

    def allocate_slots(self, request: Request, *args, **kwargs) -> KVCacheBlocks | None:
        result = self._allocate_slots(request, *args, **kwargs)
        if result is None and self.pages:
            # A capacity failure happens before allocation. Return optional
            # pages and retry once to preserve ordinary admission.
            self.pool.free_blocks(self.pages.values())
            self.pages.clear()
            result = self._allocate_slots(request, *args, **kwargs)
        return result

    def free(self, request: Request) -> None:
        prepared = self.pages.pop(request.request_id, None)
        if prepared is not None:
            self.pool.free_blocks([prepared])
        self._free(request)

    def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:
        blocks = self._pop_blocks_for_free(request)
        prepared = self.pages.pop(request.request_id, None)
        if prepared is not None:
            blocks.append(prepared)
        return blocks


class DSparkNextPageScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._draft_pages = DraftPageReservations(self.kv_cache_manager)

    def _prepare_draft_pages(self, output: SchedulerOutput) -> None:
        if output.skip_drafts or output.scheduled_new_reqs:
            return
        requests = [self.requests[rid] for rid in output.num_scheduled_tokens]
        if any(request.is_prefill_chunk for request in requests):
            return
        ids = self._draft_pages.prepare(
            requests, output.num_scheduled_tokens, self.num_lookahead_tokens, 2 * (1 + self.num_spec_tokens)
        )
        if ids:
            # SchedulerOutput is a dataclass. The MP worker RPC pickles the
            # instance, including this plugin-owned attribute.
            setattr(output, "draft_block_ids_to_prepare", ids)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        self._prepare_draft_pages(scheduler_output)
        super()._update_after_schedule(scheduler_output)


class DSparkNextPageAsyncScheduler(AsyncScheduler, DSparkNextPageScheduler):
    """Retain the standard AsyncScheduler post-schedule bookkeeping."""
