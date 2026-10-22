"""Tests for py-block-decomposition.

Conventions:
    * All ranges are half-open: [lo, hi).
    * ``apply`` adds a scalar to a range; ``apply_to_block`` does the same in
      block-index space; ``get`` reconciles an element with its block's pending
      delta.

These tests cover the awkward edges: block_size == 1, non-multiple length,
empty ranges, single-block ranges, ranges that start/end on boundaries, ranges
that span exactly one full block, and the ``apply_to_block`` complement.
"""

import unittest

from py_block_decomposition import BlockDecomposition, BlockRange


class TestConstruction(unittest.TestCase):
    def test_rejects_non_positive_block_size(self):
        with self.assertRaises(ValueError):
            BlockDecomposition([1, 2, 3], 0)
        with self.assertRaises(ValueError):
            BlockDecomposition([1, 2, 3], -1)

    def test_rejects_non_integer_block_size(self):
        with self.assertRaises(ValueError):
            BlockDecomposition([1, 2, 3], 2.0)  # type: ignore[arg-type]

    def test_accepts_empty_array(self):
        bd = BlockDecomposition([], 4)
        self.assertEqual(bd.size, 0)
        self.assertEqual(bd.num_blocks, 0)

    def test_does_not_mutate_callers_list(self):
        original = [1, 2, 3, 4]
        bd = BlockDecomposition(original, 2)
        bd.apply(0, 4, 10)
        self.assertEqual(original, [1, 2, 3, 4])


class TestBlockBounds(unittest.TestCase):
    def test_full_blocks(self):
        bd = BlockDecomposition([0, 0, 0, 0, 0, 0], 2)
        self.assertEqual(bd.num_blocks, 3)
        self.assertEqual(bd.block_bounds(0), (0, 2))
        self.assertEqual(bd.block_bounds(1), (2, 4))
        self.assertEqual(bd.block_bounds(2), (4, 6))

    def test_short_tail_block(self):
        bd = BlockDecomposition([0, 0, 0, 0, 0], 2)
        self.assertEqual(bd.num_blocks, 3)
        self.assertEqual(bd.block_bounds(2), (4, 5))

    def test_block_bounds_out_of_range(self):
        bd = BlockDecomposition([0, 0, 0], 2)
        with self.assertRaises(IndexError):
            bd.block_bounds(2)
        with self.assertRaises(IndexError):
            bd.block_bounds(-1)


class TestGet(unittest.TestCase):
    def test_get_without_delta(self):
        bd = BlockDecomposition([1, 2, 3, 4], 2)
        self.assertEqual(bd.get(0), 1)
        self.assertEqual(bd.get(3), 4)

    def test_get_with_delta(self):
        bd = BlockDecomposition([0, 0, 0, 0], 2)
        bd.apply(0, 4, 5)
        self.assertEqual(bd.get(0), 5)
        self.assertEqual(bd.get(3), 5)

    def test_get_out_of_range(self):
        bd = BlockDecomposition([0, 0], 1)
        with self.assertRaises(IndexError):
            bd.get(2)
        with self.assertRaises(IndexError):
            bd.get(-1)


class TestSplit(unittest.TestCase):
    def test_empty_range(self):
        bd = BlockDecomposition([0, 0, 0, 0], 2)
        parts = bd.split(2, 2)
        self.assertEqual(parts, BlockRange((0, 0), (0, 0), (0, 0)))

    def test_single_partial_block(self):
        bd = BlockDecomposition([0, 0, 0, 0], 2)
        parts = bd.split(0, 1)
        self.assertEqual(parts, BlockRange((0, 1), (0, 0), (0, 0)))

    def test_left_partial_then_full_then_right_partial(self):
        bd = BlockDecomposition([0] * 8, 2)
        # [1, 7): partial block 0, full blocks 1-2, partial block 3
        parts = bd.split(1, 7)
        self.assertEqual(parts.left, (0, 1))
        self.assertEqual(parts.interior, (1, 3))
        self.assertEqual(parts.right, (3, 4))

    def test_range_aligned_to_block_boundaries(self):
        bd = BlockDecomposition([0] * 8, 2)
        # [2, 6): starts and ends on boundaries
        parts = bd.split(2, 6)
        self.assertEqual(parts.left, (1, 1))
        self.assertEqual(parts.interior, (1, 3))
        self.assertEqual(parts.right, (2, 2))

    def test_range_aligned_start_partial_end(self):
        bd = BlockDecomposition([0] * 8, 2)
        # [2, 7): starts on boundary, ends mid-block
        parts = bd.split(2, 7)
        self.assertEqual(parts.left, (1, 1))
        self.assertEqual(parts.interior, (1, 3))
        self.assertEqual(parts.right, (3, 4))

    def test_range_partial_start_aligned_end(self):
        bd = BlockDecomposition([0] * 8, 2)
        # [1, 6): starts mid-block, ends on boundary
        parts = bd.split(1, 6)
        self.assertEqual(parts.left, (0, 1))
        self.assertEqual(parts.interior, (1, 3))
        self.assertEqual(parts.right, (2, 2))

    def test_split_rejects_invalid_range(self):
        bd = BlockDecomposition([0, 0, 0, 0], 2)
        with self.assertRaises(ValueError):
            bd.split(-1, 2)
        with self.assertRaises(ValueError):
            bd.split(0, 5)
        with self.assertRaises(ValueError):
            bd.split(3, 2)


class TestApply(unittest.TestCase):
    def test_apply_empty_range_noop(self):
        bd = BlockDecomposition([1, 2, 3, 4], 2)
        bd.apply(2, 2, 10)
        self.assertEqual(bd.materialize(), [1, 2, 3, 4])

    def test_apply_single_element(self):
        bd = BlockDecomposition([0, 0, 0, 0], 2)
        bd.apply(1, 2, 7)
        self.assertEqual(bd.materialize(), [0, 7, 0, 0])

    def test_apply_full_range_uses_only_deltas(self):
        bd = BlockDecomposition([0, 0, 0, 0, 0, 0], 2)
        bd.apply(0, 6, 5)
        # All deltas pending; elements still zero until materialized.
        self.assertEqual(bd.pending_delta(0), 5)
        self.assertEqual(bd.pending_delta(1), 5)
        self.assertEqual(bd.pending_delta(2), 5)
        self.assertEqual(bd.materialize(), [5, 5, 5, 5, 5, 5])

    def test_apply_left_partial_only(self):
        bd = BlockDecomposition([0, 0, 0, 0], 2)
        bd.apply(0, 1, 3)
        self.assertEqual(bd.materialize(), [3, 0, 0, 0])
        self.assertEqual(bd.pending_delta(0), 0)

    def test_apply_right_partial_only(self):
        bd = BlockDecomposition([0, 0, 0, 0], 2)
        bd.apply(3, 4, 3)
        self.assertEqual(bd.materialize(), [0, 0, 0, 3])

    def test_apply_spanning_partial_left_and_right(self):
        bd = BlockDecomposition([0, 0, 0, 0, 0, 0], 2)
        bd.apply(1, 5, 4)
        self.assertEqual(bd.materialize(), [0, 4, 4, 4, 4, 0])

    def test_apply_aligned_to_block_boundaries(self):
        bd = BlockDecomposition([0, 0, 0, 0, 0, 0], 2)
        bd.apply(2, 6, 4)
        self.assertEqual(bd.materialize(), [0, 0, 4, 4, 4, 4])

    def test_apply_after_delta_pushes_other_elements(self):
        # Apply full block, then partial update to same block: the partial must
        # settle the pending delta on the OTHER elements first.
        bd = BlockDecomposition([0, 0, 0, 0], 2)
        bd.apply(0, 2, 10)  # delta on block 0
        bd.apply(0, 1, 1)   # partial on block 0
        self.assertEqual(bd.materialize(), [11, 10, 0, 0])

    def test_apply_partial_on_right_after_interior(self):
        bd = BlockDecomposition([0, 0, 0, 0, 0, 0], 2)
        bd.apply(0, 5, 3)
        self.assertEqual(bd.materialize(), [3, 3, 3, 3, 3, 0])

    def test_apply_block_size_one(self):
        bd = BlockDecomposition([0, 0, 0, 0], 1)
        bd.apply(1, 3, 9)
        self.assertEqual(bd.materialize(), [0, 9, 9, 0])

    def test_apply_negative_value(self):
        bd = BlockDecomposition([5, 5, 5, 5], 2)
        bd.apply(0, 4, -3)
        self.assertEqual(bd.materialize(), [2, 2, 2, 2])

    def test_apply_rejects_invalid_range(self):
        bd = BlockDecomposition([0, 0, 0, 0], 2)
        with self.assertRaises(ValueError):
            bd.apply(-1, 2, 1)
        with self.assertRaises(ValueError):
            bd.apply(0, 5, 1)
        with self.assertRaises(ValueError):
            bd.apply(3, 2, 1)


class TestApplyToBlock(unittest.TestCase):
    def test_apply_to_block_single(self):
        bd = BlockDecomposition([0, 0, 0, 0], 2)
        bd.apply_to_block(1, 2, 7)
        self.assertEqual(bd.pending_delta(1), 7)
        self.assertEqual(bd.materialize(), [0, 0, 7, 7])

    def test_apply_to_block_range(self):
        bd = BlockDecomposition([0, 0, 0, 0, 0, 0], 2)
        bd.apply_to_block(1, 3, 5)
        self.assertEqual(bd.materialize(), [0, 0, 5, 5, 5, 5])

    def test_apply_to_block_empty(self):
        bd = BlockDecomposition([0, 0, 0, 0], 2)
        bd.apply_to_block(1, 1, 9)
        self.assertEqual(bd.materialize(), [0, 0, 0, 0])

    def test_apply_to_block_all(self):
        bd = BlockDecomposition([0, 0, 0, 0], 2)
        bd.apply_to_block(0, 2, 3)
        self.assertEqual(bd.materialize(), [3, 3, 3, 3])

    def test_apply_to_block_rejects_invalid(self):
        bd = BlockDecomposition([0, 0, 0, 0], 2)
        with self.assertRaises(ValueError):
            bd.apply_to_block(-1, 2, 1)
        with self.assertRaises(ValueError):
            bd.apply_to_block(0, 3, 1)
        with self.assertRaises(ValueError):
            bd.apply_to_block(2, 1, 1)


class TestMaterialize(unittest.TestCase):
    def test_materialize_does_not_mutate_internal_state(self):
        bd = BlockDecomposition([0, 0, 0, 0], 2)
        bd.apply(0, 4, 5)
        out1 = bd.materialize()
        out2 = bd.materialize()
        self.assertEqual(out1, out2)
        self.assertEqual(bd.pending_delta(0), 5)

    def test_materialize_after_partial_settle(self):
        bd = BlockDecomposition([0, 0, 0, 0], 2)
        bd.apply(0, 2, 10)
        bd.apply(0, 1, 1)
        # block 0's delta was settled to 0; block 1 still 0
        self.assertEqual(bd.pending_delta(0), 0)
        self.assertEqual(bd.materialize(), [11, 10, 0, 0])


class TestShortTail(unittest.TestCase):
    def test_short_tail_block_partial_update(self):
        # length 5, block_size 2 -> blocks [0,1,2], block 2 has 1 element
        bd = BlockDecomposition([0, 0, 0, 0, 0], 2)
        bd.apply(4, 5, 9)
        self.assertEqual(bd.materialize(), [0, 0, 0, 0, 9])

    def test_short_tail_block_full_via_apply_to_block(self):
        bd = BlockDecomposition([0, 0, 0, 0, 0], 2)
        bd.apply_to_block(2, 3, 9)
        self.assertEqual(bd.materialize(), [0, 0, 0, 0, 9])

    def test_short_tail_spanning_into_tail(self):
        bd = BlockDecomposition([0, 0, 0, 0, 0], 2)
        bd.apply(3, 5, 4)
        self.assertEqual(bd.materialize(), [0, 0, 0, 4, 4])


class TestPendingDelta(unittest.TestCase):
    def test_pending_delta_out_of_range(self):
        bd = BlockDecomposition([0, 0, 0, 0], 2)
        with self.assertRaises(IndexError):
            bd.pending_delta(2)
        with self.assertRaises(IndexError):
            bd.pending_delta(-1)

    def test_pending_delta_zero_initially(self):
        bd = BlockDecomposition([1, 2, 3, 4], 2)
        self.assertEqual(bd.pending_delta(0), 0)
        self.assertEqual(bd.pending_delta(1), 0)


class TestProperties(unittest.TestCase):
    def test_block_size_property(self):
        bd = BlockDecomposition([0, 0, 0], 5)
        self.assertEqual(bd.block_size, 5)

    def test_num_blocks_property(self):
        bd = BlockDecomposition([0, 0, 0, 0, 0, 0, 0], 3)
        self.assertEqual(bd.num_blocks, 3)

    def test_size_property(self):
        bd = BlockDecomposition([1, 2, 3, 4, 5], 2)
        self.assertEqual(bd.size, 5)


if __name__ == "__main__":
    unittest.main()
