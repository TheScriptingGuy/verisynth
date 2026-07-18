"""Acceptance tests for the deterministic metadata-skeleton inference layer
(TASK CARD 16): ``verisynth.scanner.infer_skeleton`` / ``init_from_dir``, and
the non-interactive ``verisynth init --input DIR --yes`` CLI path that uses
them.

See docs/ARCHITECTURE.md §2, §3, §7 for the metadata DSL these rules target,
and verisynth/_skeleton_infer.py for the rule implementation.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from verisynth.backbone import validate_dataset
from verisynth.engine import Engine
from verisynth.fit import fit_metadata
from verisynth.metadata import load_metadata, metadata_to_dict, parse_metadata
from verisynth.scanner import infer_skeleton, init_from_dir


# --------------------------------------------------------------------------
# Unit tests -- individual rules
# --------------------------------------------------------------------------


def test_pk_preference_id_named_wins():
    n = 50
    df = pl.DataFrame(
        {
            "widget_id": np.arange(n, dtype=np.int64),
            "sku": [f"SKU-{i:04d}" for i in range(n)],  # also unique, but not id-like
            "color": np.random.default_rng(0).choice(["red", "blue"], n),
        }
    )
    md, warnings = infer_skeleton({"widgets": df}, seed=1)
    t = md.tables["widgets"]
    assert t.primary_key == "widget_id"
    assert t.role == "root"
    assert t.rows == n


def test_fk_below_threshold_is_not_a_relation():
    parent = pl.DataFrame({"category_id": np.arange(10, dtype=np.int64)})
    # 95 valid + 5 out-of-range category_id values -> 95% coverage, below the
    # 98% threshold (Card16 rule 2) -> no relation, "items" stays a root.
    cat_vals = np.array([i % 10 for i in range(95)] + [999] * 5, dtype=np.int64)
    child = pl.DataFrame(
        {
            "item_id": np.arange(100, dtype=np.int64),
            "category_id": cat_vals,
        }
    )
    md, warnings = infer_skeleton({"categories": parent, "items": child}, seed=1)
    assert md.tables["items"].role == "root"
    assert md.tables["items"].parent is None


def test_parent_vs_reference_smallest_mean_wins():
    rng = np.random.default_rng(2)
    orders = pl.DataFrame({"order_id": np.arange(20, dtype=np.int64)})
    products = pl.DataFrame({"product_id": np.arange(5, dtype=np.int64)})
    # 2 items per order (mean children-per-parent via order_id = 2) vs.
    # heavy fan-out through 5 products (mean children-per-parent = 8).
    order_id_col = np.repeat(np.arange(20, dtype=np.int64), 2)
    n = len(order_id_col)
    product_id_col = rng.integers(0, 5, n).astype(np.int64)
    items = pl.DataFrame({"order_id": order_id_col, "product_id": product_id_col})

    md, warnings = infer_skeleton(
        {"orders": orders, "products": products, "order_items": items}, seed=1
    )
    oi = md.tables["order_items"]
    assert oi.role == "child"
    assert oi.parent == "orders"
    product_col = oi.columns["product_id"]
    assert product_col.reference == "products"
    assert product_col.distribution.kind == "zipf"
    assert product_col.distribution.params == {"a": 0.5, "n": 5}


def test_cardinality_bernoulli_vs_poisson():
    parents = pl.DataFrame({"parent_id": np.arange(50, dtype=np.int64)})

    # All-or-nothing participation -> bernoulli placeholder.
    fk_bernoulli = np.arange(0, 50, 2, dtype=np.int64)  # each parent <= 1 child
    bernoulli_child = pl.DataFrame({"parent_id": fk_bernoulli})
    md, _ = infer_skeleton({"parents": parents, "kids_b": bernoulli_child}, seed=1)
    kb = md.tables["kids_b"]
    assert kb.cardinality.kind == "bernoulli"
    assert kb.cardinality.params == {"p": 0.5}
    assert kb.child_stride == 2

    # Multiple children per parent -> poisson placeholder.
    rng = np.random.default_rng(3)
    counts = rng.integers(1, 6, 50)
    fk_poisson = np.repeat(np.arange(50, dtype=np.int64), counts)
    poisson_child = pl.DataFrame({"parent_id": fk_poisson})
    md2, _ = infer_skeleton({"parents": parents, "kids_p": poisson_child}, seed=1)
    kp = md2.tables["kids_p"]
    assert kp.cardinality.kind == "poisson"
    observed_max = int(counts.max())
    assert kp.cardinality.params["max"] == max(1, int(np.ceil(observed_max * 1.5)))
    assert kp.child_stride > kp.cardinality.params["max"]
    assert kp.child_stride & (kp.child_stride - 1) == 0


def test_high_cardinality_string_omitted_with_warning():
    n = 300
    df = pl.DataFrame(
        {
            "row_id": np.arange(n, dtype=np.int64),
            "free_text": [f"unique-value-{i}" for i in range(n)],  # 300 distinct > 200
        }
    )
    md, warnings = infer_skeleton({"logs": df}, seed=1)
    assert "free_text" not in md.tables["logs"].columns
    assert any("high-cardinality string column logs.free_text omitted" in w for w in warnings)


def test_int_low_cardinality_becomes_categorical():
    rng = np.random.default_rng(4)
    n = 500
    df = pl.DataFrame(
        {
            "row_id": np.arange(n, dtype=np.int64),
            "rating": rng.integers(1, 6, n).astype(np.int64),  # 5 distinct values
        }
    )
    md, _ = infer_skeleton({"ratings": df}, seed=1)
    rating = md.tables["ratings"].columns["rating"]
    assert rating.distribution.kind == "categorical"
    assert rating.distribution.params["categories"] == sorted(set(df["rating"].to_list()))
    probs = rating.distribution.params["probs"]
    assert len(probs) == len(rating.distribution.params["categories"])
    assert abs(sum(probs) - 1.0) < 1e-9


def test_null_rate_captured():
    n = 400
    rng = np.random.default_rng(5)
    vals = rng.normal(0, 1, n).tolist()
    for i in range(40):  # 10% null
        vals[i] = None
    df = pl.DataFrame({"row_id": np.arange(n, dtype=np.int64), "score": vals})
    md, _ = infer_skeleton({"scores": df}, seed=1)
    col = md.tables["scores"].columns["score"]
    assert col.null_rate == pytest.approx(0.1, abs=1e-3)


def test_temporal_chain_recovered_with_parent_anchor():
    n_sessions = 200
    base = datetime(2023, 1, 1)
    rng = np.random.default_rng(6)
    created = [base + timedelta(seconds=int(s)) for s in rng.integers(0, 1_000_000, n_sessions)]
    sessions = pl.DataFrame(
        {"session_id": np.arange(n_sessions, dtype=np.int64), "created_at": created}
    )

    reps = 3
    session_id_col = np.repeat(np.arange(n_sessions, dtype=np.int64), reps)
    n = len(session_id_col)
    delay_a = rng.uniform(10, 50, n)
    delay_b = rng.uniform(100, 300, n)
    delay_c = rng.uniform(1000, 3000, n)
    created_rep = np.repeat(np.array([c.timestamp() for c in created]), reps)
    t_a = [datetime.fromtimestamp(created_rep[i] + delay_a[i]) for i in range(n)]
    t_b = [datetime.fromtimestamp(created_rep[i] + delay_a[i] + delay_b[i]) for i in range(n)]
    t_c = [
        datetime.fromtimestamp(created_rep[i] + delay_a[i] + delay_b[i] + delay_c[i])
        for i in range(n)
    ]
    events = pl.DataFrame(
        {"session_id": session_id_col, "t_a": t_a, "t_b": t_b, "t_c": t_c}
    )

    md, _ = infer_skeleton({"sessions": sessions, "events": events}, seed=1)
    ev = md.tables["events"]
    assert ev.role == "child" and ev.parent == "sessions"
    assert ev.columns["t_a"].temporal.anchor == "sessions.created_at"
    assert ev.columns["t_b"].temporal.anchor == "t_a"
    assert ev.columns["t_c"].temporal.anchor == "t_b"


def test_copula_proposed_and_not_proposed():
    rng = np.random.default_rng(7)
    n = 2000
    mean = [0, 0]
    cov_high = [[1.0, 0.6], [0.6, 1.0]]
    xy = rng.multivariate_normal(mean, cov_high, n)
    cov_low = [[1.0, 0.1], [0.1, 1.0]]
    zw = rng.multivariate_normal(mean, cov_low, n)
    df = pl.DataFrame(
        {
            "row_id": np.arange(n, dtype=np.int64),
            "x": xy[:, 0],
            "y": xy[:, 1],
            "z": zw[:, 0],
            "w": zw[:, 1],
        }
    )
    md, _ = infer_skeleton({"metrics": df}, seed=1)
    t = md.tables["metrics"]
    grouped_cols = {c for cop in t.copulas for c in cop.columns}
    assert {"x", "y"} <= grouped_cols
    assert "z" not in grouped_cols and "w" not in grouped_cols


def test_deterministic_two_runs_identical():
    rng = np.random.default_rng(8)
    orders = pl.DataFrame({"order_id": np.arange(30, dtype=np.int64)})
    items = pl.DataFrame(
        {
            "order_id": rng.integers(0, 30, 90).astype(np.int64),
            "price": rng.lognormal(2, 1, 90),
        }
    )
    frames = {"orders": orders, "order_items": items}
    md1, w1 = infer_skeleton(frames, seed=42)
    md2, w2 = infer_skeleton(frames, seed=42)
    assert metadata_to_dict(md1) == metadata_to_dict(md2)
    assert w1 == w2


def test_parent_as_pk_rule():
    n_orders = 200
    orders = pl.DataFrame({"order_id": np.arange(n_orders, dtype=np.int64)})
    # Each shipment's own natural unique column ("order_id") is itself a
    # foreign key into orders' PK -> shipments should become a bernoulli
    # child of orders via a synthetic PK (Card16 delta rule 1).
    shipped_orders = np.arange(0, n_orders, 2, dtype=np.int64)  # 50% coverage-ish
    shipments = pl.DataFrame(
        {
            "order_id": shipped_orders,
            "carrier": np.random.default_rng(9).choice(["ups", "dhl"], len(shipped_orders)),
        }
    )
    md, warnings = infer_skeleton({"orders": orders, "shipments": shipments}, seed=1)
    sh = md.tables["shipments"]
    assert sh.role == "child"
    assert sh.parent == "orders"
    assert sh.primary_key not in ("order_id",)
    assert sh.columns["order_id"].generator == "parent_key"
    assert sh.columns[sh.primary_key].generator == "key"
    assert sh.cardinality.kind == "bernoulli"
    assert sh.child_stride == 2
    assert any("parent-as-pk rule" in w for w in warnings)


# --------------------------------------------------------------------------
# Olist round-trip (flagship)
# --------------------------------------------------------------------------

_OLIST_DIR = Path(__file__).resolve().parent.parent / "examples" / "olist" / "data"


def _load_olist_frames() -> dict[str, pl.DataFrame]:
    return {f.stem: pl.read_parquet(f) for f in sorted(_OLIST_DIR.glob("*.parquet"))}


@pytest.fixture(scope="module")
def olist_frames():
    return _load_olist_frames()


def test_olist_round_trip(olist_frames):
    md, warnings = infer_skeleton(
        olist_frames,
        seed=20240817,
        sources=[("crm", "crm_*"), ("inventory", "inv_*"), ("shop", "*")],
    )

    expected_parents = {
        "customers": "crm_contacts",
        "orders": "customers",
        "order_items": "orders",
        "order_payments": "orders",
        "order_reviews": "orders",
        "crm_tickets": "crm_contacts",
        "inv_shipments": "orders",
    }
    for child, parent in expected_parents.items():
        assert md.tables[child].parent == parent, child

    assert md.tables["crm_contacts"].role == "root"
    assert md.tables["inv_products"].role == "root"

    product_col = md.tables["order_items"].columns["product_id"]
    assert product_col.reference == "inv_products"

    assert md.tables["customers"].cardinality.kind == "bernoulli"

    orders = md.tables["orders"]
    assert orders.columns["order_approved_at"].temporal.anchor == "order_purchase_timestamp"
    assert (
        orders.columns["order_delivered_customer_date"].temporal.anchor
        == "order_delivered_carrier_date"
    )

    basket_cols = {c for cop in md.tables["order_items"].copulas for c in cop.columns}
    assert {"price", "freight_value"} <= basket_cols

    payment_cols = {c for cop in md.tables["order_payments"].copulas for c in cop.columns}
    assert {"payment_installments", "payment_value"} <= payment_cols

    # Round-trips through the YAML-safe dict representation.
    assert parse_metadata(metadata_to_dict(md)) is not None


# --------------------------------------------------------------------------
# End-to-end: init -> fit -> generate -> validate
# --------------------------------------------------------------------------


def test_end_to_end_init_fit_generate_validate(olist_frames, tmp_path):
    skeleton, _warnings = infer_skeleton(
        olist_frames,
        seed=20240817,
        sources=[("crm", "crm_*"), ("inventory", "inv_*"), ("shop", "*")],
    )
    fitted = fit_metadata(olist_frames, skeleton)

    out_dir = tmp_path / "out"
    Engine(fitted, seed=fitted.seed).generate(str(out_dir), num_partitions=1)
    violations = validate_dataset(fitted, str(out_dir))
    assert violations == []


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_init_yes_with_sources(tmp_path):
    out = tmp_path / "skel.yaml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "verisynth.cli",
            "init",
            "--input",
            str(_OLIST_DIR),
            "-o",
            str(out),
            "--seed",
            "20240817",
            "--yes",
            "--source",
            "crm=crm_*",
            "--source",
            "inventory=inv_*",
            "--source",
            "shop=*",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    md = load_metadata(out)
    assert md.tables["crm_contacts"].source == "crm"
    assert md.tables["inv_products"].source == "inventory"
    assert md.tables["orders"].source == "shop"
    # Inferred-structure summary: one line per table.
    assert "role=" in result.stdout
    assert "pk=" in result.stdout
