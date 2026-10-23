# py-block-decomposition

Partitions an array into fixed-size blocks and tracks a pending add per block, so that range updates over full blocks are O(number of blocks) instead of O(number of elements).

## Usage

```python
from py_block_decomposition import BlockDecomposition

data = [0, 0, 0, 0, 0, 0]
bd = BlockDecomposition(data, block_size=2)

# Add 5 to elements [0, 6). Interior full blocks only touch a pending delta.
bd.apply(0, 6, 5)
assert bd.get(0) == 5
assert bd.materialize() == [5, 5, 5, 5, 5, 5]

# Partial range [1, 5) adds 4 to the middle, leaving the ends alone.
bd.apply(1, 5, 4)
assert bd.materialize() == [5, 9, 9, 9, 9, 5]

# Block-index API: add 7 to every element in block index 1.
bd.apply_to_block(1, 2, 7)
```

Exported names: `BlockDecomposition`, `BlockRange`.

## Why this exists

The problem is the boring but common one: you have a flat array and a workload that repeatedly adds a scalar to contiguous ranges. A naive loop is O(range length) per update. A segment tree with lazy propagation solves it generally but is easy to get wrong and harder to audit.

This library picks the middle ground: fixed-size blocks, one pending integer delta per block, add-only updates. Bulk updates over full interior blocks touch only the delta table. Partial blocks at the edges of a range are settled element-by-element, which costs O(block_size) at most twice per update. The trade-off is that reads must reconcile an element with its block's pending delta, and there is no support for non-additive updates or sub-range queries beyond what `materialize()` produces. That is deliberate: the surface is small and the failure modes are obvious.

## The awkward edge

A partial update that touches only part of a block cannot be represented by the block's delta alone, because the delta applies to the whole block. The implementation handles this by first pushing the block's existing delta onto all of its elements, zeroing the delta, and then applying the partial update directly. This is O(block_size) and happens at most twice per `apply` call. If your workload is dominated by tiny partial updates, pick a smaller `block_size`; if it is dominated by wide full-block updates, pick a larger one.
