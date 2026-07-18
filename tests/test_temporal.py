"""Acceptance tests for verisynth.temporal (delay propagation along the
event-anchor DAG). See docs/ARCHITECTURE.md §5 and TASK CARD 6.
"""

from __future__ import annotations

import numpy as np
import pytest

from verisynth import kernels, temporal
from verisynth.distributions import make_delay_ppf
from verisynth.metadata import (
    ColumnSpec,
    DistributionSpec,
    MetadataError,
    TableSpec,
    TemporalSpec,
)

SEED = 42


def _ts(name: str, anchor: str, delay: DistributionSpec) -> ColumnSpec:
    return ColumnSpec(name=name, type="timestamp", temporal=TemporalSpec(anchor=anchor, delay=delay))


def _plain(name: str, generator: str | None = None, type_: str = "timestamp") -> ColumnSpec:
    return ColumnSpec(name=name, type=type_, generator=generator)


EXP_DELAY = DistributionSpec(kind="exponential", params={"rate": 1.0e-6})
LOGN_DELAY = DistributionSpec(kind="lognormal", params={"mu": 11.5, "sigma": 0.6})


def _orders_table(declare_reversed: bool = False) -> TableSpec:
    """Model orders from ARCHITECTURE.md §2: ordered_at anchored on
    customers.signup_at (cross-table), shipped_at anchored on ordered_at
    (same-table).
    """
    order_id = _plain("order_id", generator="key", type_="int64")
    customer_id = _plain("customer_id", generator="parent_key", type_="int64")
    ordered_at = _ts("ordered_at", "customers.signup_at", EXP_DELAY)
    shipped_at = _ts("shipped_at", "ordered_at", LOGN_DELAY)

    if declare_reversed:
        columns = {
            "order_id": order_id,
            "customer_id": customer_id,
            "shipped_at": shipped_at,
            "ordered_at": ordered_at,
        }
    else:
        columns = {
            "order_id": order_id,
            "customer_id": customer_id,
            "ordered_at": ordered_at,
            "shipped_at": shipped_at,
        }

    return TableSpec(
        name="orders",
        role="child",
        columns=columns,
        primary_key="order_id",
        parent="customers",
        cardinality=None,
        child_stride=64,
    )


# --------------------------------------------------------------------------
# 1. Ordering
# --------------------------------------------------------------------------


def test_order_puts_ordered_at_before_shipped_at_regardless_of_declaration_order():
    for declare_reversed in (False, True):
        table = _orders_table(declare_reversed=declare_reversed)
        order = temporal.order_temporal_columns(table)
        assert order.index("ordered_at") < order.index("shipped_at")
        assert set(order) == {"ordered_at", "shipped_at"}


def test_order_three_link_chain():
    a = _ts("a", "root.anchor", EXP_DELAY)  # cross-table, no dep
    b = _ts("b", "a", EXP_DELAY)
    c = _ts("c", "b", EXP_DELAY)
    table = TableSpec(
        name="t",
        role="child",
        columns={"c": c, "b": b, "a": a},
        primary_key="a",
        parent="root",
        cardinality=None,
        child_stride=4,
    )
    order = temporal.order_temporal_columns(table)
    assert order.index("a") < order.index("b") < order.index("c")
    assert set(order) == {"a", "b", "c"}


def test_order_stable_among_independent_columns():
    # Two columns each anchored on a cross-table column: no intra-table
    # dependency between them, so declaration order should be preserved.
    x = _ts("x", "root.anchor1", EXP_DELAY)
    y = _ts("y", "root.anchor2", EXP_DELAY)
    table = TableSpec(
        name="t",
        role="child",
        columns={"x": x, "y": y},
        primary_key="x",
        parent="root",
        cardinality=None,
        child_stride=4,
    )
    assert temporal.order_temporal_columns(table) == ["x", "y"]

    table_rev = TableSpec(
        name="t",
        role="child",
        columns={"y": y, "x": x},
        primary_key="x",
        parent="root",
        cardinality=None,
        child_stride=4,
    )
    assert temporal.order_temporal_columns(table_rev) == ["y", "x"]


def test_order_cycle_raises_metadata_error():
    p = _ts("p", "q", EXP_DELAY)
    q = _ts("q", "p", EXP_DELAY)
    table = TableSpec(
        name="t",
        role="root",
        columns={"p": p, "q": q},
        primary_key="p",
        rows=10,
    )
    with pytest.raises(MetadataError, match="cycle"):
        temporal.order_temporal_columns(table)


# --------------------------------------------------------------------------
# 2. Correctness
# --------------------------------------------------------------------------


def _hand_compute(table: TableSpec, cname: str, row_keys: np.ndarray, anchor_values, anchor_null_mask):
    col = table.columns[cname]
    ppf = make_delay_ppf(col.temporal.delay)
    u = kernels.keyed_uniforms(SEED, f"{table.name}.{cname}.__delay__", row_keys)
    delay_seconds = np.maximum(ppf(u), 0.0)
    delay_us = np.trunc(delay_seconds * 1e6).astype(np.int64)
    values = anchor_values + delay_us
    null_mask = anchor_null_mask.copy()
    values = values.copy()
    values[null_mask] = 0
    return values, null_mask


def test_correctness_hand_recomputation():
    table = _orders_table()
    row_keys = np.arange(5000, dtype=np.uint64)
    anchor_values = np.full(5000, 1_700_000_000_000_000, dtype=np.int64)
    anchor_null_mask = np.zeros(5000, dtype=np.bool_)

    result = temporal.propagate(
        SEED, table, row_keys, {"customers.signup_at": (anchor_values, anchor_null_mask)}
    )

    exp_ordered_values, exp_ordered_null = _hand_compute(
        table, "ordered_at", row_keys, anchor_values, anchor_null_mask
    )
    np.testing.assert_array_equal(result["ordered_at"][0], exp_ordered_values)
    np.testing.assert_array_equal(result["ordered_at"][1], exp_ordered_null)

    exp_shipped_values, exp_shipped_null = _hand_compute(
        table, "shipped_at", row_keys, exp_ordered_values, exp_ordered_null
    )
    np.testing.assert_array_equal(result["shipped_at"][0], exp_shipped_values)
    np.testing.assert_array_equal(result["shipped_at"][1], exp_shipped_null)

    assert np.all(result["shipped_at"][0] >= result["ordered_at"][0])


# --------------------------------------------------------------------------
# 3. Null propagation
# --------------------------------------------------------------------------


def test_null_propagation():
    table = _orders_table()
    row_keys = np.arange(1000, dtype=np.uint64)
    anchor_values = np.full(1000, 1_700_000_000_000_000, dtype=np.int64)
    anchor_null_mask = np.zeros(1000, dtype=np.bool_)
    null_positions = np.array([3, 17, 500, 999])
    anchor_null_mask[null_positions] = True

    result = temporal.propagate(
        SEED, table, row_keys, {"customers.signup_at": (anchor_values, anchor_null_mask)}
    )

    for cname in ("ordered_at", "shipped_at"):
        values, null_mask = result[cname]
        np.testing.assert_array_equal(null_mask, anchor_null_mask)
        assert np.all(values[null_positions] == 0)
        assert np.all(~np.isnan(values.astype(np.float64)))  # sanity: no NaNs sneak in
        # non-null positions must not be zero (delay + huge anchor value)
        non_null = ~anchor_null_mask
        assert np.all(values[non_null] != 0)


# --------------------------------------------------------------------------
# 4. Determinism + partition invariance
# --------------------------------------------------------------------------


def test_determinism_and_partition_invariance():
    table = _orders_table()
    full_keys = np.arange(1000, dtype=np.uint64)
    full_anchor_values = (1_700_000_000_000_000 + full_keys.astype(np.int64) * 1000)
    full_anchor_null_mask = np.zeros(1000, dtype=np.bool_)

    full_result = temporal.propagate(
        SEED, table, full_keys, {"customers.signup_at": (full_anchor_values, full_anchor_null_mask)}
    )

    slice_keys = full_keys[100:200]
    slice_anchor_values = full_anchor_values[100:200]
    slice_anchor_null_mask = full_anchor_null_mask[100:200]

    slice_result = temporal.propagate(
        SEED,
        table,
        slice_keys,
        {"customers.signup_at": (slice_anchor_values, slice_anchor_null_mask)},
    )

    direct_keys = np.arange(100, 200, dtype=np.uint64)
    direct_result = temporal.propagate(
        SEED,
        table,
        direct_keys,
        {"customers.signup_at": (slice_anchor_values, slice_anchor_null_mask)},
    )

    for cname in ("ordered_at", "shipped_at"):
        np.testing.assert_array_equal(full_result[cname][0][100:200], slice_result[cname][0])
        np.testing.assert_array_equal(full_result[cname][1][100:200], slice_result[cname][1])
        np.testing.assert_array_equal(slice_result[cname][0], direct_result[cname][0])
        np.testing.assert_array_equal(slice_result[cname][1], direct_result[cname][1])

    # Repeated call, same inputs -> identical outputs.
    repeat_result = temporal.propagate(
        SEED, table, full_keys, {"customers.signup_at": (full_anchor_values, full_anchor_null_mask)}
    )
    for cname in ("ordered_at", "shipped_at"):
        np.testing.assert_array_equal(full_result[cname][0], repeat_result[cname][0])
        np.testing.assert_array_equal(full_result[cname][1], repeat_result[cname][1])


# --------------------------------------------------------------------------
# 5. Missing anchor
# --------------------------------------------------------------------------


def test_missing_anchor_key_raises_key_error():
    table = _orders_table()
    row_keys = np.arange(10, dtype=np.uint64)
    with pytest.raises(KeyError, match="customers.signup_at"):
        temporal.propagate(SEED, table, row_keys, {})


# --------------------------------------------------------------------------
# 6. Empty row_keys
# --------------------------------------------------------------------------


def test_empty_row_keys():
    table = _orders_table()
    row_keys = np.array([], dtype=np.uint64)
    anchor_values = np.array([], dtype=np.int64)
    anchor_null_mask = np.array([], dtype=np.bool_)

    result = temporal.propagate(
        SEED, table, row_keys, {"customers.signup_at": (anchor_values, anchor_null_mask)}
    )

    assert set(result) == {"ordered_at", "shipped_at"}
    for cname in ("ordered_at", "shipped_at"):
        values, null_mask = result[cname]
        assert values.shape == (0,)
        assert null_mask.shape == (0,)
        assert values.dtype == np.int64
        assert null_mask.dtype == np.bool_
