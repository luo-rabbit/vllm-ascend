# SPDX-License-Identifier: Apache-2.0
"""Overlap DSpark next-page clearing with GLM5.3 target projections."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch
import torch_npu
from vllm.logger import logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .zero_window_layout import batches_for_components, projection_windows

# Measured on A5 EP16, 32 verification tokens, SWA scheduler block size 128.
NUM_REQUESTS = 4
TOKENS_PER_REQUEST = 8
TOKENS = NUM_REQUESTS * TOKENS_PER_REQUEST
NO_BLOCK_ID = -1
_ZERO_GRID = (16,)
_PROJECTIONS = 59
_NOMINAL_BATCHES = 44  # Prior single-ID work was about 1.75 ms.
_active_capture: ContextVar["ZeroWindowPlan | None"] = ContextVar("zero_window_capture", default=None)


@triton.jit(do_not_specialize=["first_tile", "end_tile"])
def _zero_kv_window_kernel(
    addrs,
    strides,
    page_sizes,
    segments,
    offsets,
    block_id_ptr,
    first_tile,
    end_tile,
    TILE_WORDS: tl.constexpr,
):
    block_id = tl.load(block_id_ptr)
    if block_id >= 0:
        columns = tl.arange(0, TILE_WORDS).to(tl.int64)
        zeros = tl.zeros((TILE_WORDS,), dtype=tl.int32)
        for tile in range(first_tile + tl.program_id(0), end_tile, tl.num_programs(0)):
            segment = tl.load(segments + tile)
            start = tl.load(offsets + tile).to(tl.int64)
            base = tl.cast(tl.load(addrs + segment), tl.pointer_type(tl.int32))
            stride = tl.load(strides + segment)
            page_size = tl.load(page_sizes + segment)
            words = start + columns
            tl.store(base + block_id * stride + words, zeros, mask=words < page_size)


class ZeroWindowPlan:
    def __init__(self, metas: list, device: torch.device) -> None:
        metas = [meta for meta in metas if meta is not None]
        components = [(meta[2].cpu().tolist(), meta[4]) for meta in metas]
        _, layouts = batches_for_components(components, _NOMINAL_BATCHES)
        count = sum(len(layout[2]) for layout in layouts)
        windows = projection_windows(count, _PROJECTIONS)
        self.batches: dict[int, tuple] = {}
        self.seen = 0
        self.recorded = 0
        if not windows:
            logger.warning("DSpark zero windows unavailable: %d batches for %d windows", count, _PROJECTIONS)
            return
        self.stream = torch.npu.Stream(device=device)
        self.block_id = torch.full((1,), NO_BLOCK_ID, device=device, dtype=torch.int64)
        self.drain = torch.npu.Event()
        ordinals = iter(windows)
        for meta, (segment_ids, offsets, ranges) in zip(metas, layouts, strict=True):
            if not ranges:
                continue
            seg_tensor = torch.tensor(segment_ids, dtype=torch.int32, device=device)
            offset_tensor = torch.tensor(offsets, dtype=torch.int32, device=device)
            for first, end, _ in ranges:
                args = (*meta[:3], seg_tensor, offset_tensor, self.block_id, first, end)
                self.batches[next(ordinals)] = (torch.npu.Event(), args, meta[4])
        # Compile before capture. The sentinel disables all KV writes.
        for _, args, tile_words in self.batches.values():
            _zero_kv_window_kernel[_ZERO_GRID](*args, TILE_WORDS=tile_words)
        logger.info("DSpark zero windows planned: %d batches across %d projections", count, _PROJECTIONS)

    @contextmanager
    def capture(self) -> Iterator[None]:
        self.seen = self.recorded = 0
        token = _active_capture.set(self)
        try:
            yield
        finally:
            _active_capture.reset(token)
            if self.recorded:
                self.drain.record(self.stream)
                torch.npu.current_stream().wait_event(self.drain)

    def before_projection(self) -> None:
        batch = self.batches.get(self.seen)
        self.seen += 1
        if batch is None:
            return
        event, args, tile_words = batch
        event.record()
        self.stream.wait_event(event)
        with torch.npu.stream(self.stream):
            _zero_kv_window_kernel[_ZERO_GRID](*args, TILE_WORDS=tile_words)
        self.recorded += 1

    def is_complete(self) -> bool:
        return bool(self.batches) and self.seen == _PROJECTIONS and self.recorded == len(self.batches)


def is_window_projection(weight_scale: torch.Tensor) -> bool:
    return weight_scale.shape == (96, 24896, 2)


def _window_quant_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    pertoken_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype,
    group_size: int,
) -> torch.Tensor:
    # The Python capture state is consulted at runtime, outside Dynamo tracing.
    plan = _active_capture.get()
    if plan is not None and x.shape == (TOKENS, 6144):
        plan.before_projection()
    return torch_npu.npu_quant_matmul(
        x,
        weight,
        weight_scale,
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale=pertoken_scale,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        bias=bias,
        output_dtype=output_dtype,
        group_sizes=[1, 1, group_size],
    )


def _window_quant_matmul_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    pertoken_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype,
    group_size: int,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight_scale.shape[1]), dtype=output_dtype)


direct_register_custom_op(
    op_name="dspark_window_quant_matmul",
    op_func=_window_quant_matmul,
    fake_impl=_window_quant_matmul_fake,
    dispatch_key="PrivateUse1",
)
