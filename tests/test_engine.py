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
from verisynth.metadata import load_metadata, parse_metadata
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


# --------------------------------------------------------------------------
# 7. Multi-source dataset: generator: parent:{column} inheritance,
#    bernoulli cardinality, cross-source `source:` output routing, and a
#    cross-table temporal anchor fed by an inherited timestamp.
#    See TASK CARD 11 and docs/ARCHITECTURE.md §8.
# --------------------------------------------------------------------------

MULTI_SOURCE_SEED = 7
N_CONTACTS = 2000


def _multi_source_metadata_dict(rows: int = N_CONTACTS) -> dict:
    return {
        "version": 1,
        "seed": MULTI_SOURCE_SEED,
        "tables": {
            "crm_contacts": {
                "role": "root",
                "rows": rows,
                "primary_key": "contact_id",
                "source": "crm",
                "columns": {
                    "contact_id": {"type": "int64", "generator": "key"},
                    "state": {
                        "type": "string",
                        "distribution": {
                            "kind": "categorical",
                            "categories": ["A", "B", "C"],
                            "probs": [0.5, 0.3, 0.2],
                        },
                    },
                    "created_at": {
                        "type": "timestamp",
                        "distribution": {
                            "kind": "datetime_uniform",
                            "start": "2022-01-01T00:00:00",
                            "end": "2023-01-01T00:00:00",
                        },
                    },
                },
            },
            "customers": {
                "role": "child",
                "parent": "crm_contacts",
                "cardinality": {"kind": "bernoulli", "p": 0.7},
                "child_stride": 2,
                "primary_key": "customer_id",
                "source": "shop",
                "columns": {
                    "customer_id": {"type": "int64", "generator": "key"},
                    "contact_id": {"type": "int64", "generator": "parent_key"},
                    "state": {"type": "string", "generator": "parent:state"},
                    "created_at": {"type": "timestamp", "generator": "parent:created_at"},
                },
            },
            "orders": {
                "role": "child",
                "parent": "customers",
                "cardinality": {"kind": "poisson", "lam": 2.0, "max": 15},
                "child_stride": 16,
                "primary_key": "order_id",
                "columns": {
                    "order_id": {"type": "int64", "generator": "key"},
                    "customer_id": {"type": "int64", "generator": "parent_key"},
                    "ordered_at": {
                        "type": "timestamp",
                        "temporal": {
                            "anchor": "customers.created_at",
                            "delay": {"kind": "exponential", "rate": 1.0e-6},
                        },
                    },
                },
            },
        },
    }


def _multi_source_metadata(rows: int = N_CONTACTS):
    return parse_metadata(_multi_source_metadata_dict(rows))


def test_multi_source_inheritance_and_cardinality():
    md = _multi_source_metadata(N_CONTACTS)
    eng = Engine(md, seed=MULTI_SOURCE_SEED)
    tables = eng.generate_partition(0, 1)

    contacts = tables["crm_contacts"]
    customers = tables["customers"]
    orders = tables["orders"]

    contact_id = contacts.column("contact_id").to_numpy(zero_copy_only=False)
    contact_state = contacts.column("state").to_numpy(zero_copy_only=False)
    contact_created_at = contacts.column("created_at").to_numpy(zero_copy_only=False)
    state_by_contact = dict(zip(contact_id.tolist(), contact_state.tolist()))
    created_by_contact = dict(zip(contact_id.tolist(), contact_created_at.tolist()))

    cust_contact_id = customers.column("contact_id").to_numpy(zero_copy_only=False)
    cust_state = customers.column("state").to_numpy(zero_copy_only=False)
    cust_created_at = customers.column("created_at").to_numpy(zero_copy_only=False)

    # (1) inherited state equals the parent's state row-for-row (joined via
    # contact_id).
    expected_state = np.array([state_by_contact[cid] for cid in cust_contact_id.tolist()])
    assert np.array_equal(cust_state, expected_state)

    # (2) inherited timestamp equals parent's exactly.
    expected_created = np.array(
        [created_by_contact[cid] for cid in cust_contact_id.tolist()], dtype=cust_created_at.dtype
    )
    assert np.array_equal(cust_created_at, expected_created)

    # (3) customers row count ~= 0.7 * contacts (binomial tolerance); every
    # contact_id unique (bernoulli cardinality is at most 1 per parent).
    n_customers = customers.num_rows
    expected_n = 0.7 * N_CONTACTS
    std = (N_CONTACTS * 0.7 * 0.3) ** 0.5
    assert abs(n_customers - expected_n) < 5 * std
    assert len(np.unique(cust_contact_id)) == len(cust_contact_id)

    # (6) orders' temporal anchor ordering: order ts >= inherited created_at.
    customer_id = customers.column("customer_id").to_numpy(zero_copy_only=False)
    created_by_customer = dict(zip(customer_id.tolist(), cust_created_at.tolist()))
    orders_customer_id = orders.column("customer_id").to_numpy(zero_copy_only=False)
    ordered_at = orders.column("ordered_at").to_numpy(zero_copy_only=False)
    order_anchor = np.array(
        [created_by_customer[cid] for cid in orders_customer_id.tolist()], dtype=ordered_at.dtype
    )
    ordered_valid = ~pa.array(ordered_at).is_null().to_numpy(zero_copy_only=False)
    assert np.all(ordered_at[ordered_valid] >= order_anchor[ordered_valid])


def test_multi_source_partition_invariance():
    md1 = _multi_source_metadata(N_CONTACTS)
    single = Engine(md1, seed=MULTI_SOURCE_SEED).generate_partition(0, 1)

    md3 = _multi_source_metadata(N_CONTACTS)
    eng3 = Engine(md3, seed=MULTI_SOURCE_SEED)
    parts = [eng3.generate_partition(p, 3) for p in range(3)]

    for name in single:
        concatenated = pa.concat_tables([parts[p][name] for p in range(3)])
        assert concatenated.equals(single[name]), f"table {name} not partition-invariant"


def test_multi_source_generate_writes_source_paths_and_validates(tmp_path):
    md = _multi_source_metadata(N_CONTACTS)
    eng = Engine(md, seed=MULTI_SOURCE_SEED)
    out_dir = tmp_path / "out"
    eng.generate(str(out_dir), num_partitions=2)

    assert (out_dir / "crm" / "crm_contacts" / "part-00000.parquet").exists()
    assert (out_dir / "crm" / "crm_contacts" / "part-00001.parquet").exists()
    assert (out_dir / "shop" / "customers" / "part-00000.parquet").exists()
    assert (out_dir / "shop" / "customers" / "part-00001.parquet").exists()
    # orders has no `source:` -> unchanged top-level layout.
    assert (out_dir / "orders" / "part-00000.parquet").exists()

    assert validate_dataset(md, out_dir) == []
