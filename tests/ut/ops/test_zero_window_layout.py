# SPDX-License-Identifier: Apache-2.0
import unittest

from vllm_ascend.ops.triton.zero_window_layout import batches_for_components, projection_windows


class WindowLayoutTest(unittest.TestCase):
    def test_covers_each_valid_word_and_charges_the_partial_tile(self):
        components = [([1, 17, 128, 0, 29], 16), ([81, 7, 48], 16)]
        budget, layouts = batches_for_components(components, nominal_batches=9)
        for (page_words, tile_words), (segments, offsets, batches) in zip(
            components, layouts, strict=True
        ):
            covered = [[] for _ in page_words]
            position = 0
            for first, end, words in batches:
                self.assertEqual(first, position)
                self.assertLessEqual(words, budget)
                batch_words = 0
                for segment, offset in zip(segments[first:end], offsets[first:end], strict=True):
                    span = range(offset, min(offset + tile_words, page_words[segment]))
                    covered[segment].extend(span)
                    batch_words += len(span)
                self.assertEqual(words, batch_words)
                position = end
            self.assertEqual(position, len(segments))
            self.assertEqual(covered, [list(range(words)) for words in page_words])

    def test_budget_can_fit_the_largest_tile(self):
        budget, layouts = batches_for_components([([29], 32), ([1], 4)], 44)
        self.assertEqual(budget, 29)
        self.assertEqual([batch for _, _, batches in layouts for batch in batches],
                         [(0, 1, 29), (0, 1, 1)])

    def test_window_ordinals_are_unique_and_spaced(self):
        windows = projection_windows(46, 59)
        self.assertEqual(len(windows), 46)
        self.assertEqual(windows, sorted(set(windows)))
        self.assertGreater(windows[-1], 50)
        self.assertLess(windows[-1], 59)
        self.assertEqual(projection_windows(59, 59), list(range(59)))
        self.assertEqual(projection_windows(60, 59), [])
        self.assertEqual(projection_windows(0, 59), [])
