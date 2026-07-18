"""Partition planner: partition-by-root ranges and child cardinality/keys.

See docs/ARCHITECTURE.md §3 (normative). Root tables are split into
contiguous ranges by partition; child row_keys are derived deterministically
from parent row_keys via ``parent_row_key * child_stride + j``.
"""

from __future__ import annotations

import numpy as np

from . import kernels
from .distributions import make_marginal
from .metadata import CardinalitySpec


def root_range(rows: int, partition: int, num_partitions: int) -> tuple[int, int]:
    """Return the half-open ``[lo, hi)`` row-index range owned by ``partition``."""
    if num_partitions < 1:
        raise ValueError(f"num_partitions must be >= 1 (got {num_partitions!r})")
    if not (0 <= partition < num_partitions):
        raise ValueError(
            f"partition must satisfy 0 <= partition < num_partitions "
            f"(got partition={partition!r}, num_partitions={num_partitions!r})"
        )
    if rows < 0:
        raise ValueError(f"rows must be >= 0 (got {rows!r})")

    lo = (rows * partition) // num_partitions
    hi = (rows * (partition + 1)) // num_partitions
    return lo, hi


def root_keys(rows: int, partition: int, num_partitions: int) -> np.ndarray:
    """Return the root row_keys (uint64) owned by ``partition``."""
    lo, hi = root_range(rows, partition, num_partitions)
    return np.arange(lo, hi, dtype=np.uint64)


def child_counts(
    seed: int, child_table: str, spec: CardinalitySpec, parent_keys: np.ndarray
) -> np.ndarray:
    """Sample per-parent child counts (int64), deterministic given ``parent_keys``."""
    if spec.kind == "fixed":
        n = int(spec.params["n"])
        cap = n
        counts = np.full(len(parent_keys), n, dtype=np.int64)
        return np.clip(counts, 0, cap)

    cap = int(spec.params["max"])
    u = kernels.keyed_uniforms(seed, f"{child_table}.__cardinality__", parent_keys)
    counts = make_marginal(spec).ppf(u).astype(np.int64)
    return np.clip(counts, 0, cap)


def expand_children(
    parent_keys: np.ndarray, counts: np.ndarray, child_stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Expand parent keys/counts into child_keys and their parent positions."""
    if child_stride <= 0:
        raise ValueError(f"child_stride must be > 0 (got {child_stride!r})")

    counts = np.asarray(counts, dtype=np.int64)
    m = len(counts)
    total = int(counts.sum()) if m else 0

    if total == 0:
        return (
            np.empty(0, dtype=np.uint64),
            np.empty(0, dtype=np.int64),
        )

    parent_pos = np.repeat(np.arange(m, dtype=np.int64), counts)

    cumulative_offsets = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
    j = np.arange(total, dtype=np.int64) - np.repeat(cumulative_offsets, counts)

    with np.errstate(over="ignore"):
        child_keys = parent_keys[parent_pos] * np.uint64(child_stride) + j.astype(np.uint64)

    return child_keys, parent_pos
