"""Acceptance tests for verisynth.engine (Engine orchestration).

See docs/ARCHITECTURE.md §3-§6 (normative) and TASK CARD 8.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from verisynth.backbone import ParquetBackbone, validate_dataset
from verisynth.engine import Engine
from verisynth.metadata import load_metadata
from verisynth.partition import child_counts, expand_children, root_keys

REPO_ROOT = Path(__file__).resolve().parent.parent
RETAIL_YAML = REPO_ROOT / "examples" / "retail.yaml"

SEED = 42


def _small_metadata(rows: int = 800):
    md = load_metadata(RETAIL_YAML)
    md.tables["customers"].rows = rows
    return md


# --------------------------------------------------------------------------
# 1. Shapes / schema
# --------------------------------------------------------------------------


def test_shapes_and_schema():
    md = _small_metadata(800)
    eng = Engine(md, seed=SEED)
    tables = eng.generate_partition(0, 1)

    customers = tables["customers"]
    orders = tables["orders"]

    assert customers.num_rows == 800

    rk = root_keys(800, 0, 1)
    counts = child_counts(SEED, "orders", md.tables["orders"].cardinality, rk)
    assert orders.num_rows == int(counts.sum())

    expected_customer_cols = ["customer_id", "region", "age", "income", "signup_at"]
    assert customers.schema.names == expected_customer_cols
    assert customers.schema.field("customer_id").type == pa.int64()
    assert customers.schema.field("region").type == pa.string()
    assert customers.schema.field("age").type == pa.int64()
    assert customers.schema.field("income").type == pa.float64()
    assert customers.schema.field("signup_at").type == pa.timestamp("us")

    expected_order_cols = [
        "order_id",
        "customer_id",
        "order_total",
        "ordered_at",
        "shipped_at",
        "order_total_eur",  # derived, appended last
    ]
    assert orders.schema.names == expected_order_cols
    assert orders.schema.field("order_id").type == pa.int64()
    assert orders.schema.field("customer_id").type == pa.int64()
    assert orders.schema.field("order_total").type == pa.float64()
    assert orders.schema.field("ordered_at").type == pa.timestamp("us")
    assert orders.schema.field("shipped_at").type == pa.timestamp("us")
    assert orders.schema.field("order_total_eur").type == pa.float64()


# --------------------------------------------------------------------------
# 2. Determinism
# --------------------------------------------------------------------------


def test_determinism():
    md_a = _small_metadata(800)
    md_b = _small_metadata(800)
    tables_a = Engine(md_a, seed=SEED).generate_partition(0, 1)
    tables_b = Engine(md_b, seed=SEED).generate_partition(0, 1)

    for name in tables_a:
        assert tables_a[name].equals(tables_b[name]), f"table {name} not deterministic"


# --------------------------------------------------------------------------
# 3. Partition invariance
# --------------------------------------------------------------------------


def test_partition_invariance():
    md1 = _small_metadata(800)
    eng1 = Engine(md1, seed=SEED)
    single = eng1.generate_partition(0, 1)

    md3 = _small_metadata(800)
    eng3 = Engine(md3, seed=SEED)
    parts = [eng3.generate_partition(p, 3) for p in range(3)]

    for name in single:
        concatenated = pa.concat_tables([parts[p][name] for p in range(3)])
        assert concatenated.equals(single[name]), f"table {name} not partition-invariant"


# --------------------------------------------------------------------------
# 4. Statistical spot-checks
# --------------------------------------------------------------------------


def test_statistical_spot_checks():
    md = _small_metadata(5000)
    eng = Engine(md, seed=SEED)
    tables = eng.generate_partition(0, 1)
    customers = tables["customers"]
    orders = tables["orders"]

    age = customers.column("age").to_numpy(zero_copy_only=False)
    income = customers.column("income").to_numpy(zero_copy_only=False)

    assert age.min() >= 18
    assert age.max() <= 95
    assert np.all(age == age.astype(np.int64))

    corr = np.corrcoef(age.astype(np.float64), income.astype(np.float64))[0, 1]
    assert corr > 0.35, f"expected age/income corr > 0.35, got {corr}"

    region = customers.column("region").to_numpy(zero_copy_only=False)
    n = len(region)
    freqs = {
        "NA": np.sum(region == "NA") / n,
        "EU": np.sum(region == "EU") / n,
        "APAC": np.sum(region == "APAC") / n,
    }
    expected = {"NA": 0.5, "EU": 0.3, "APAC": 0.2}
    for k, exp in expected.items():
        assert abs(freqs[k] - exp) < 0.03, f"region {k} freq {freqs[k]} vs expected {exp}"

    order_total = orders.column("order_total")
    null_frac = order_total.null_count / len(order_total)
    assert 0.003 <= null_frac <= 0.03, f"order_total null frac {null_frac} out of range"

    # shipped_at >= ordered_at >= signup_at of owning customer, wherever non-null
    customer_id = customers.column("customer_id").to_numpy(zero_copy_only=False)
    signup_at = customers.column("signup_at").to_numpy(zero_copy_only=False)
    signup_by_id = dict(zip(customer_id.tolist(), signup_at.tolist()))

    orders_customer_id = orders.column("customer_id").to_numpy(zero_copy_only=False)
    ordered_at = orders.column("ordered_at").to_numpy(zero_copy_only=False)
    shipped_at = orders.column("shipped_at").to_numpy(zero_copy_only=False)
    order_signup_at = np.array(
        [signup_by_id[cid] for cid in orders_customer_id.tolist()], dtype=ordered_at.dtype
    )

    ordered_valid = ~pa.array(ordered_at).is_null().to_numpy(zero_copy_only=False)
    shipped_valid = ~pa.array(shipped_at).is_null().to_numpy(zero_copy_only=False)

    assert np.all(ordered_at[ordered_valid] >= order_signup_at[ordered_valid])
    both_valid = ordered_valid & shipped_valid
    assert np.all(shipped_at[both_valid] >= ordered_at[both_valid])

    # order_total_eur ~ order_total * 0.92 (allclose, nulls aligned)
    order_total_np = order_total.to_numpy(zero_copy_only=False)
    order_total_eur = orders.column("order_total_eur").to_numpy(zero_copy_only=False)
    null_mask = np.isnan(order_total_np)
    assert np.array_equal(null_mask, np.isnan(order_total_eur))
    valid = ~null_mask
    np.testing.assert_allclose(order_total_eur[valid], order_total_np[valid] * 0.92)


# --------------------------------------------------------------------------
# 5. Empty partitions
# --------------------------------------------------------------------------


def test_empty_partitions():
    md_single = _small_metadata(3)
    single = Engine(md_single, seed=SEED).generate_partition(0, 1)

    md_p = _small_metadata(3)
    eng_p = Engine(md_p, seed=SEED)
    parts = [eng_p.generate_partition(p, 8) for p in range(8)]

    empty_seen = False
    for name in single:
        for p in range(8):
            if parts[p][name].num_rows == 0:
                empty_seen = True
                assert parts[p][name].schema == single[name].schema

    assert empty_seen, "expected at least one empty partition with rows=3, P=8"

    for name in single:
        concatenated = pa.concat_tables([parts[p][name] for p in range(8)])
        assert concatenated.equals(single[name])


# --------------------------------------------------------------------------
# 6. validate_dataset
# --------------------------------------------------------------------------


def test_validate_dataset_ok_then_corrupted(tmp_path):
    md = _small_metadata(800)
    eng = Engine(md, seed=SEED)
    out_dir = tmp_path / "out"
    eng.generate(str(out_dir), num_partitions=3)

    assert validate_dataset(md, out_dir) == []

    # Corrupt: inject a bogus orders row with an unknown customer_id.
    orders_schema = eng.generate_partition(0, 1)["orders"].schema
    bogus = pa.table(
        {
            "order_id": pa.array([999_999_999], type=pa.int64()),
            "customer_id": pa.array([-1], type=pa.int64()),
            "order_total": pa.array([1.23], type=pa.float64()),
            "ordered_at": pa.array([0], type=pa.timestamp("us")),
            "shipped_at": pa.array([0], type=pa.timestamp("us")),
            "order_total_eur": pa.array([1.13], type=pa.float64()),
        },
        schema=orders_schema,
    )

    backbone = ParquetBackbone(out_dir)
    backbone.write_partition("orders", bogus, 99)

    violations = validate_dataset(md, out_dir)
    assert violations, "expected FK violation after corrupting orders"
    assert any("customer_id" in v or "orders" in v for v in violations)
