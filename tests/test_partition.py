"""Acceptance tests for verisynth.partition (partition planner + child keys)."""

from __future__ import annotations

import numpy as np
import pytest

from verisynth.metadata import CardinalitySpec
from verisynth.partition import (
    child_counts,
    expand_children,
    root_keys,
    root_range,
)


# --------------------------------------------------------------------------
# 1. Exact partition coverage
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rows,P", [(10, 3), (1000, 7), (5, 8), (0, 2), (1000, 1)])
def test_partition_coverage(rows, P):
    pieces = [root_keys(rows, p, P) for p in range(P)]
    concatenated = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.uint64)
    assert np.array_equal(concatenated, np.arange(rows, dtype=np.uint64))

    ranges = [root_range(rows, p, P) for p in range(P)]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == rows
    for (lo, hi) in ranges:
        assert lo <= hi
    for i in range(len(ranges) - 1):
        assert ranges[i][1] == ranges[i + 1][0]


def test_root_range_invalid_partition():
    with pytest.raises(ValueError):
        root_range(10, -1, 3)
    with pytest.raises(ValueError):
        root_range(10, 3, 3)
    with pytest.raises(ValueError):
        root_range(10, 0, 0)
    with pytest.raises(ValueError):
        root_range(10, 0, -1)
    with pytest.raises(ValueError):
        root_range(-1, 0, 3)


# --------------------------------------------------------------------------
# 2. child_counts
# --------------------------------------------------------------------------


def test_child_counts_determinism():
    spec = CardinalitySpec(kind="poisson", params={"lam": 4.2, "max": 63})
    parent_keys = np.arange(500, dtype=np.uint64)
    c1 = child_counts(1, "orders", spec, parent_keys)
    c2 = child_counts(1, "orders", spec, parent_keys)
    assert np.array_equal(c1, c2)
    assert c1.dtype == np.int64


def test_child_counts_poisson_bounds_and_mean():
    lam = 4.2
    cap = 63
    spec = CardinalitySpec(kind="poisson", params={"lam": lam, "max": cap})
    parent_keys = np.arange(20_000, dtype=np.uint64)
    counts = child_counts(1, "orders", spec, parent_keys)
    assert np.all(counts >= 0)
    assert np.all(counts <= cap)
    mean = counts.mean()
    assert abs(mean - lam) < 0.03 * lam


def test_child_counts_fixed():
    spec = CardinalitySpec(kind="fixed", params={"n": 7})
    parent_keys = np.arange(100, dtype=np.uint64)
    counts = child_counts(1, "items", spec, parent_keys)
    assert np.all(counts == 7)
    assert counts.dtype == np.int64


def test_child_counts_uniform_int():
    low, high, cap = 2, 10, 8
    spec = CardinalitySpec(kind="uniform_int", params={"low": low, "high": high, "max": cap})
    parent_keys = np.arange(5_000, dtype=np.uint64)
    counts = child_counts(1, "lines", spec, parent_keys)
    assert np.all(counts >= low)
    assert np.all(counts <= min(high, cap))


def test_child_counts_bernoulli_values_mean_and_determinism():
    p = 0.6
    spec = CardinalitySpec(kind="bernoulli", params={"p": p})
    parent_keys = np.arange(20_000, dtype=np.uint64)
    counts1 = child_counts(1, "accounts", spec, parent_keys)
    counts2 = child_counts(1, "accounts", spec, parent_keys)

    assert counts1.dtype == np.int64
    assert set(np.unique(counts1)) <= {0, 1}
    assert abs(counts1.mean() - p) < 0.01
    assert np.array_equal(counts1, counts2)


def test_child_counts_bernoulli_partition_invariance():
    spec = CardinalitySpec(kind="bernoulli", params={"p": 0.6})
    all_keys = np.arange(1000, dtype=np.uint64)
    sliced = all_keys[300:400]
    direct = np.arange(300, 400, dtype=np.uint64)

    c_sliced = child_counts(7, "accounts", spec, sliced)
    c_direct = child_counts(7, "accounts", spec, direct)
    assert np.array_equal(c_sliced, c_direct)


# --------------------------------------------------------------------------
# 3. Partition invariance of child_counts
# --------------------------------------------------------------------------


def test_child_counts_partition_invariance():
    spec = CardinalitySpec(kind="poisson", params={"lam": 4.2, "max": 63})
    all_keys = np.arange(1000, dtype=np.uint64)
    sliced = all_keys[300:400]
    direct = np.arange(300, 400, dtype=np.uint64)
    assert np.array_equal(sliced, direct)

    c_sliced = child_counts(7, "orders", spec, sliced)
    c_direct = child_counts(7, "orders", spec, direct)
    assert np.array_equal(c_sliced, c_direct)


# --------------------------------------------------------------------------
# 4. expand_children
# --------------------------------------------------------------------------


def test_expand_children_hand_check():
    parent_keys = np.array([5, 9], dtype=np.uint64)
    counts = np.array([2, 0], dtype=np.int64)
    child_keys, parent_pos = expand_children(parent_keys, counts, 64)
    assert np.array_equal(child_keys, np.array([320, 321], dtype=np.uint64))
    assert np.array_equal(parent_pos, np.array([0, 0], dtype=np.int64))


def test_expand_children_empty():
    parent_keys = np.array([5, 9], dtype=np.uint64)
    counts = np.array([0, 0], dtype=np.int64)
    child_keys, parent_pos = expand_children(parent_keys, counts, 64)
    assert len(child_keys) == 0
    assert len(parent_pos) == 0
    assert child_keys.dtype == np.uint64
    assert parent_pos.dtype == np.int64


def test_expand_children_global_uniqueness_and_parent_mapping():
    rng = np.random.default_rng(0)
    m = 10_000
    parent_keys = rng.choice(10_000_000, size=m, replace=False).astype(np.uint64)

    spec = CardinalitySpec(kind="poisson", params={"lam": 4.2, "max": 63})
    counts = child_counts(3, "orders", spec, parent_keys)

    stride = 64
    child_keys, parent_pos = expand_children(parent_keys, counts, stride)

    assert len(child_keys) == int(counts.sum())
    assert len(np.unique(child_keys)) == len(child_keys)

    # Spot-check parent mapping via integer division by stride.
    assert np.array_equal(child_keys // np.uint64(stride), parent_keys[parent_pos])


# --------------------------------------------------------------------------
# 5. Two-level composition
# --------------------------------------------------------------------------


def test_two_level_composition_unique_grandchildren():
    rng = np.random.default_rng(1)
    m = 2_000
    parent_keys = rng.choice(1_000_000, size=m, replace=False).astype(np.uint64)

    child_spec = CardinalitySpec(kind="poisson", params={"lam": 4.2, "max": 63})
    child_counts_arr = child_counts(5, "orders", child_spec, parent_keys)
    child_keys, _ = expand_children(parent_keys, child_counts_arr, 64)

    grandchild_spec = CardinalitySpec(kind="poisson", params={"lam": 1.5, "max": 31})
    grandchild_counts_arr = child_counts(5, "items", grandchild_spec, child_keys)
    grandchild_keys, grandchild_parent_pos = expand_children(
        child_keys, grandchild_counts_arr, 32
    )

    assert len(grandchild_keys) == int(grandchild_counts_arr.sum())
    assert len(np.unique(grandchild_keys)) == len(grandchild_keys)
    assert np.array_equal(
        grandchild_keys // np.uint64(32), child_keys[grandchild_parent_pos]
    )


# --------------------------------------------------------------------------
# 6. End-to-end partition invariance
# --------------------------------------------------------------------------


def test_end_to_end_partition_invariance():
    rows = 1000
    seed = 42
    table = "orders"
    spec = CardinalitySpec(kind="poisson", params={"lam": 4.2, "max": 63})
    stride = 64

    rk1 = root_keys(rows, 0, 1)
    counts1 = child_counts(seed, table, spec, rk1)
    child_keys1, _ = expand_children(rk1, counts1, stride)

    pieces = []
    for p in range(4):
        rk = root_keys(rows, p, 4)
        counts = child_counts(seed, table, spec, rk)
        ck, _ = expand_children(rk, counts, stride)
        pieces.append(ck)
    child_keys4 = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.uint64)

    assert np.array_equal(child_keys1, child_keys4)
