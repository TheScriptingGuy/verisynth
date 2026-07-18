"""Acceptance tests for verisynth.scanner (structure detection from real data).

Covers column profiling, PK/FK detection, cardinality suggestion
(bernoulli / fixed / poisson / uniform_int + child_stride), and the report
renderers backing `verisynth scan`.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from verisynth.scanner import (
    _cardinality_profile,
    rank_parent_relations,
    render_report,
    report_to_dict,
    scan_directory,
)

N_CUSTOMERS = 300


def _customers_df(rng: np.random.Generator) -> pl.DataFrame:
    base = datetime(2022, 1, 1)
    return pl.DataFrame(
        {
            "customer_id": np.arange(N_CUSTOMERS, dtype=np.int64),
            "country": rng.choice(["NL", "BE", "DE"], size=N_CUSTOMERS, p=[0.5, 0.3, 0.2]),
            "signup_at": [base + timedelta(hours=int(h)) for h in rng.integers(0, 10_000, N_CUSTOMERS)],
            "spend": np.exp(rng.normal(4.0, 0.5, N_CUSTOMERS)),  # positive -> lognormal
            "balance": rng.normal(0.0, 10.0, N_CUSTOMERS),  # signed -> normal
            "vip": rng.random(N_CUSTOMERS) < 0.2,
        }
    )


def _orders_df(rng: np.random.Generator) -> pl.DataFrame:
    counts = rng.poisson(1.5, N_CUSTOMERS)
    fk = np.repeat(np.arange(N_CUSTOMERS, dtype=np.int64), counts)
    n = len(fk)
    df = pl.DataFrame(
        {
            "order_id": np.arange(n, dtype=np.int64),
            "customer_id": fk,
            "amount": np.exp(rng.normal(3.0, 0.7, n)),
            "status": rng.choice(["paid", "open"], size=n, p=[0.8, 0.2]),
        }
    )
    mask = pl.Series([i % 10 == 0 for i in range(n)])
    return df.with_columns(pl.when(mask).then(None).otherwise(pl.col("amount")).alias("amount"))


@pytest.fixture()
def data_dir(tmp_path):
    rng = np.random.default_rng(7)
    _customers_df(rng).write_parquet(tmp_path / "customers.parquet")
    _orders_df(rng).write_parquet(tmp_path / "orders.parquet")
    return tmp_path


def test_column_profiles_and_pk(data_dir):
    report = scan_directory(data_dir)
    cust = report.tables["customers"]

    assert cust.rows == N_CUSTOMERS
    assert cust.pk == "customer_id"
    assert cust.columns["customer_id"].unique

    assert cust.columns["customer_id"].type == "int64"
    assert cust.columns["country"].type == "string"
    assert cust.columns["signup_at"].type == "timestamp"
    assert cust.columns["spend"].type == "float64"
    assert cust.columns["vip"].type == "bool"

    orders = report.tables["orders"]
    assert orders.pk == "order_id"
    amount = orders.columns["amount"]
    assert 0.05 < amount.null_rate < 0.15
    assert not amount.unique


def test_distribution_suggestions(data_dir):
    report = scan_directory(data_dir)
    cust = report.tables["customers"]

    country = cust.columns["country"].suggestion
    assert country["kind"] == "categorical"
    assert set(country["categories"]) == {"NL", "BE", "DE"}
    assert abs(sum(country["probs"]) - 1.0) <= 1e-6

    assert cust.columns["signup_at"].suggestion["kind"] == "datetime_uniform"
    assert cust.columns["spend"].suggestion["kind"] == "lognormal"
    assert cust.columns["balance"].suggestion["kind"] == "normal"
    assert cust.columns["vip"].suggestion["kind"] == "categorical"


def test_fk_relation_and_poisson_cardinality(data_dir):
    report = scan_directory(data_dir)
    rels = report.relations_of("orders")
    assert len(rels) == 1
    r = rels[0]

    assert (r.parent, r.parent_key, r.child_column) == ("customers", "customer_id", "customer_id")
    assert r.coverage == 1.0
    assert r.cardinality["kind"] == "poisson"
    assert r.cardinality["lam"] == pytest.approx(1.5, abs=0.3)
    # child_stride: smallest power of two strictly above the observed max.
    assert r.child_stride > r.max_children
    assert r.child_stride & (r.child_stride - 1) == 0

    # customers has no inbound relation.
    assert report.relations_of("customers") == []


def test_no_relation_without_value_containment(tmp_path):
    pl.DataFrame({"customer_id": np.arange(50, dtype=np.int64)}).write_parquet(
        tmp_path / "customers.parquet"
    )
    # Name matches the parent key, but the values don't exist in the parent.
    pl.DataFrame(
        {
            "order_id": np.arange(100, dtype=np.int64),
            "customer_id": np.arange(1000, 1100, dtype=np.int64),
        }
    ).write_parquet(tmp_path / "orders.parquet")

    report = scan_directory(tmp_path)
    assert report.relations == []


def test_cardinality_kinds():
    parent = pl.DataFrame({"pk": np.arange(10, dtype=np.int64)})

    def child(counts):
        fk = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        return pl.DataFrame({"pk": fk})

    # At most one child each -> bernoulli with p = participation rate.
    spec, stride, mean, max_c = _cardinality_profile(
        child([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]), "pk", parent, "pk"
    )
    assert spec == {"kind": "bernoulli", "p": 0.5}
    assert (stride, max_c) == (2, 1)

    # Exactly two children each -> fixed n=2, stride 4.
    spec, stride, _, _ = _cardinality_profile(child([2] * 10), "pk", parent, "pk")
    assert spec == {"kind": "fixed", "n": 2}
    assert stride == 4

    # Wildly overdispersed counts -> uniform_int over the observed range.
    spec, stride, _, _ = _cardinality_profile(
        child([0, 10, 0, 10, 0, 10, 0, 10, 0, 10]), "pk", parent, "pk"
    )
    assert spec == {"kind": "uniform_int", "low": 0, "high": 10, "max": 10}
    assert stride == 16


def test_parent_ranking_prefers_name_stem(data_dir, tmp_path):
    rng = np.random.default_rng(1)
    # order_items references both orders (true parent) and a products dimension.
    orders = pl.DataFrame({"order_id": np.arange(100, dtype=np.int64)})
    products = pl.DataFrame({"product_id": np.arange(10, dtype=np.int64)})
    items = pl.DataFrame(
        {
            "order_id": rng.integers(0, 100, 300),
            "product_id": rng.integers(0, 10, 300),
        }
    )
    d = tmp_path / "ranked"
    d.mkdir()
    orders.write_parquet(d / "orders.parquet")
    products.write_parquet(d / "products.parquet")
    items.write_parquet(d / "order_items.parquet")

    report = scan_directory(d)
    rels = rank_parent_relations("order_items", report.relations_of("order_items"))
    assert [r.parent for r in rels] == ["orders", "products"]


def test_csv_loading(tmp_path):
    pl.DataFrame(
        {"thing_id": [1, 2, 3], "label": ["a", "b", "a"]}
    ).write_csv(tmp_path / "things.csv")
    report = scan_directory(tmp_path)
    t = report.tables["things"]
    assert t.pk == "thing_id"
    assert t.columns["label"].type == "string"


def test_scan_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_directory(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        scan_directory(tmp_path)  # exists but empty


def test_report_dict_and_render(data_dir):
    report = scan_directory(data_dir)
    d = report_to_dict(report)
    assert d["tables"]["orders"]["primary_key"] == "order_id"
    assert d["relations"][0]["parent"] == "customers"

    text = render_report(report)
    assert "customers" in text
    assert "fk -> customers.customer_id" in text
    assert "1--N" in text
