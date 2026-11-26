# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence


def batches_for_components(
    components: Sequence[tuple[Sequence[int], int]],
    nominal_batches: int,
) -> tuple[int, list[tuple[list[int], list[int], list[tuple[int, int, int]]]]]:
    """Pack valid tiles against a shared int32-word budget across components."""
    total_words = sum(sum(sizes) for sizes, _ in components)
    largest_tile = max(
        (min(size, tile_words) for sizes, tile_words in components for size in sizes),
        default=0,
    )
    budget = max(largest_tile, (total_words + nominal_batches - 1) // nominal_batches)
    layouts = []
    for sizes, tile_words in components:
        segments: list[int] = []
        offsets: list[int] = []
        batches: list[tuple[int, int, int]] = []
        first = words = 0
        for segment, size in enumerate(sizes):
            for offset in range(0, size, tile_words):
                valid_words = min(tile_words, size - offset)
                if words + valid_words > budget:
                    batches.append((first, len(segments), words))
                    first, words = len(segments), 0
                segments.append(segment)
                offsets.append(offset)
                words += valid_words
        if words:
            batches.append((first, len(segments), words))
        layouts.append((segments, offsets, batches))
    return budget, layouts


def projection_windows(num_batches: int, num_projections: int) -> list[int]:
    """Place at most one batch at each projection, spread over the replay."""
    if not 0 < num_batches <= num_projections:
        return []
    return [i * num_projections // num_batches for i in range(num_batches)]
