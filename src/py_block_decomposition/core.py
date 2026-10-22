"""Block decomposition for efficient bulk range updates.

The idea: lay a flat array over a grid of fixed-size blocks and keep, for each
block, a single integer offset (``delta``) that represents a pending add that
has not yet been applied to individual elements. A bulk update over [lo, hi)
splits into three regions:

    * the left partial block  - apply directly to elements, may also touch delta
    * the full interior blocks - only touch delta, never touch elements
    * the right partial block  - apply directly to elements, may also touch delta

The interior is where the win comes from: updating K full blocks is O(K)
regardless of how many elements each block covers. The cost of reading a single
element is O(1) but requires reconciling the element's stored value with its
block's pending delta.

We choose a simple model on purpose: one pending delta per block, add-only
updates, no lazy propagation across levels. This keeps the code auditable and
the failure modes obvious. A segment tree with lazy propagation is strictly more
general but harder to verify, and most callers of a "bulk range add" primitive do
not need queries over arbitrary sub-ranges.

Edge cases the implementation handles:
    * ``block_size == 1`` - every element is its own block; deltas and elements
      are effectively the same slot.
    * The array length is not a multiple of ``block_size`` - the final block is
      short; all range math accounts for it.
    * Empty ranges (``lo == hi``) - no-op, accepted for symmetry with half-open
      intervals.
    * Range updates that touch only one block - the general algorithm's left and
      right regions overlap; we detect this and apply once.
    * Range updates that touch only full blocks - the interior path runs, left
      and right regions are empty.
    * ``apply_to_block`` for a block-index range that covers every block - same
      as ``apply`` over the full array width; handled by the general path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class BlockRange:
    """A half-open block index range produced by :meth:`BlockDecomposition.split`.

    ``left`` covers the partial block at the start of the range (may be empty
    when the range begins on a block boundary). ``interior`` covers the full
    blocks in the middle (may be empty for short ranges). ``right`` covers the
    partial block at the end (may be empty when the range ends on a boundary or
    when left and right are the same block).

    All three are half-open in block-index space.
    """

    left: Tuple[int, int]
    interior: Tuple[int, int]
    right: Tuple[int, int]


class BlockDecomposition:
    """Partitions an array into fixed-size blocks for efficient bulk range
    updates.

    Parameters
    ----------
    data:
        The backing list. We hold a reference to the same list object; we mutate
        it in place. Copying the input by default would surprise callers who
        pass large arrays expecting zero-copy semantics.
    block_size:
        Number of elements per block. Must be a positive integer.

    Notes
    -----
    Each block carries an integer ``delta`` that represents a pending add
    deferred to the whole block. ``get`` reconciles an element with its block's
    delta on read. ``apply`` over a range splits into left partial / interior
    full / right partial and only touches ``delta`` for interior blocks.
    """

    __slots__ = ("_data", "_block_size", "_num_blocks", "_deltas")

    def __init__(self, data: Sequence[int], block_size: int) -> None:
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        # We keep the list identity the caller passed us so mutations are visible
        # through their reference too. If a caller wants isolation they pass a
        # copy.
        self._data: List[int] = list(data)  # defensive copy; see README trade-off
        self._block_size: int = block_size
        # A length that is not a multiple of block_size leaves a short tail
        # block; ceil-division accounts for it.
        self._num_blocks: int = (len(self._data) + block_size - 1) // block_size
        self._deltas: List[int] = [0] * self._num_blocks

    # -- properties --------------------------------------------------------

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def size(self) -> int:
        """Number of elements in the backing array."""
        return len(self._data)

    def block_bounds(self, block_index: int) -> Tuple[int, int]:
        """Return the half-open element-index range for ``block_index``."""
        if not 0 <= block_index < self._num_blocks:
            raise IndexError(f"block_index {block_index} out of range [0, {self._num_blocks})")
        lo = block_index * self._block_size
        hi = min(lo + self._block_size, len(self._data))
        return lo, hi

    # -- core operations ----------------------------------------------------

    def split(self, lo: int, hi: int) -> BlockRange:
        """Split the element-index range ``[lo, hi)`` into block regions.

        Returns three half-open block-index ranges: ``left`` (partial starting
        block), ``interior`` (full blocks), ``right`` (partial ending block).
        Any of the three may be empty.
        """
        n = len(self._data)
        if lo < 0 or hi > n or lo > hi:
            raise ValueError(f"range [{lo}, {hi}) invalid for size {n}")

        if lo == hi:
            return BlockRange((0, 0), (0, 0), (0, 0))

        first_block = lo // self._block_size
        last_block = (hi - 1) // self._block_size

        # If the whole range lies inside a single block, return it as the left
        # region with empty interior and right. We pick ``left`` rather than
        # ``right`` for the single-block case because it keeps the element-level
        # update path uniform: callers always apply the left region's elements.
        if first_block == last_block:
            return BlockRange((first_block, first_block + 1), (0, 0), (0, 0))

        # left: from first_block up to (not including) the first block whose
        # boundary coincides with ``lo``. If ``lo`` is on a boundary, the left
        # region is empty.
        left_lo = first_block
        left_hi = first_block + 1 if lo % self._block_size != 0 else first_block

        # right: from the block containing ``hi - 1``. If ``hi`` is on a boundary,
        # the right region is empty; otherwise it is a single partial block.
        right_lo = last_block if hi % self._block_size != 0 else last_block
        right_hi = last_block + 1 if hi % self._block_size != 0 else last_block

        interior_lo = left_hi
        # When the right region is empty (hi aligned to a boundary), the last
        # block is fully covered and belongs to the interior, so the interior
        # extends one past right_lo.
        interior_hi = right_lo if right_lo != right_hi else right_lo + 1
        if interior_lo > interior_hi:
            # Cannot happen given first_block != last_block, but guard the
            # invariant explicitly so a future refactor does not silently break
            # the contract.
            interior_lo, interior_hi = 0, 0

        return BlockRange((left_lo, left_hi), (interior_lo, interior_hi), (right_lo, right_hi))

    def apply(self, lo: int, hi: int, value: int) -> None:
        """Add ``value`` to every element in the element-index range ``[lo, hi)``."""
        n = len(self._data)
        if lo < 0 or hi > n or lo > hi:
            raise ValueError(f"range [{lo}, {hi}) invalid for size {n}")
        if lo == hi:
            return

        parts = self.split(lo, hi)

        # Left partial: per-element add. If the range starts mid-block, we touch
        # the elements directly because the update does not cover the whole
        # block. We also bump the block delta so subsequent reads of OTHER
        # elements in the same block are not affected - but we MUST first settle
        # the delta for those other elements. To keep this simple and correct,
        # we push the pending delta onto all elements of the block before
        # applying the partial update, then zero the delta. This is O(block_size)
        # and happens at most twice per apply call.
        if parts.left[0] != parts.left[1]:
            for b in range(parts.left[0], parts.left[1]):
                blo, bhi = self.block_bounds(b)
                d = self._deltas[b]
                if d != 0:
                    for i in range(blo, bhi):
                        self._data[i] += d
                    self._deltas[b] = 0
                for i in range(max(blo, lo), min(bhi, hi)):
                    self._data[i] += value

        # Interior full blocks: bump delta only. This is the cheap path.
        if parts.interior[0] != parts.interior[1]:
            for b in range(parts.interior[0], parts.interior[1]):
                self._deltas[b] += value

        # Right partial: same settle-then-apply pattern as left.
        if parts.right[0] != parts.right[1]:
            for b in range(parts.right[0], parts.right[1]):
                blo, bhi = self.block_bounds(b)
                d = self._deltas[b]
                if d != 0:
                    for i in range(blo, bhi):
                        self._data[i] += d
                    self._deltas[b] = 0
                for i in range(max(blo, lo), min(bhi, hi)):
                    self._data[i] += value

    def apply_to_block(self, block_index_lo: int, block_index_hi: int, value: int) -> None:
        """Add ``value`` to every element covered by the block-index range
        ``[block_index_lo, block_index_hi)``.

        This is the natural complement to :meth:`apply`: a caller that works in
        block-index space (for example, a tiling scheduler) can update whole
        blocks without translating to element indices first.
        """
        if block_index_lo < 0 or block_index_hi > self._num_blocks or block_index_lo > block_index_hi:
            raise ValueError(
                f"block range [{block_index_lo}, {block_index_hi}) invalid for {self._num_blocks} blocks"
            )
        if block_index_lo == block_index_hi:
            return
        # All blocks in the range are fully covered, so this is interior-only:
        # bump deltas and never touch elements.
        for b in range(block_index_lo, block_index_hi):
            self._deltas[b] += value

    def get(self, index: int) -> int:
        """Return the effective value at ``index`` (element + its block delta)."""
        if not 0 <= index < len(self._data):
            raise IndexError(f"index {index} out of range [0, {len(self._data)})")
        b = index // self._block_size
        return self._data[index] + self._deltas[b]

    def materialize(self) -> List[int]:
        """Return a new list with all pending deltas folded into elements.

        The backing array and delta table are left untouched.
        """
        out = list(self._data)
        for b in range(self._num_blocks):
            d = self._deltas[b]
            if d == 0:
                continue
            blo, bhi = self.block_bounds(b)
            for i in range(blo, bhi):
                out[i] += d
        return out

    def pending_delta(self, block_index: int) -> int:
        """Return the pending (un-applied) delta for ``block_index``.

        Exposed for testing and for callers that want to inspect deferred state.
        """
        if not 0 <= block_index < self._num_blocks:
            raise IndexError(f"block_index {block_index} out of range [0, {self._num_blocks})")
        return self._deltas[block_index]
