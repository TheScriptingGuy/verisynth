"""Acceptance tests for verisynth.scanner (structure detection from real data).

Covers column profiling, PK/FK detection, cardinality suggestion
(bernoulli / fixed / poisson / uniform_int + child_stride), and the report
renderers backing `verisynth scan`.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
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


# --------------------------------------------------------------------------
# JSON / XML loading
# --------------------------------------------------------------------------

N_JSON_CUSTOMERS = 20


def test_json_loading(tmp_path):
    customers = [
        {"customer_id": i, "state": ["NL", "BE", "DE"][i % 3]} for i in range(N_JSON_CUSTOMERS)
    ]
    (tmp_path / "customers.json").write_text(json.dumps(customers))

    lines = []
    for i in range(50):
        cust = i % N_JSON_CUSTOMERS
        lines.append(
            json.dumps(
                {
                    "order_id": i,
                    "customer_id": cust,
                    "amount": 10.0 + i,
                    "created_at": f"2022-01-{(i % 28) + 1:02d}T00:00:00",
                }
            )
        )
    (tmp_path / "orders.jsonl").write_text("\n".join(lines))

    report = scan_directory(tmp_path)
    assert set(report.tables) == {"customers", "orders"}

    cust = report.tables["customers"]
    assert cust.pk == "customer_id"
    assert cust.columns["customer_id"].type == "int64"
    assert cust.columns["state"].type == "string"

    orders = report.tables["orders"]
    assert orders.pk == "order_id"
    assert orders.columns["order_id"].type == "int64"
    assert orders.columns["amount"].type == "float64"
    assert orders.columns["created_at"].type == "timestamp"

    rels = report.relations_of("orders")
    assert len(rels) == 1
    assert (rels[0].parent, rels[0].parent_key, rels[0].child_column) == (
        "customers",
        "customer_id",
        "customer_id",
    )


def test_json_nested_struct_flattening(tmp_path):
    records = [
        {"order_id": i, "customer": {"customer_id": i % 5, "state": "NL"}} for i in range(10)
    ]
    (tmp_path / "orders.json").write_text(json.dumps(records))

    report = scan_directory(tmp_path)
    orders = report.tables["orders"]
    assert "customer_id" in orders.columns
    assert "state" in orders.columns
    assert "customer" not in orders.columns

    # Collision: a top-level `id` field plus a nested struct field also named
    # `id` -> the struct's field falls back to `{struct_column}_{field}`.
    collide = [{"id": i, "nested": {"id": i * 2, "label": "x"}} for i in range(5)]
    d2 = tmp_path / "collide"
    d2.mkdir()
    (d2 / "things.json").write_text(json.dumps(collide))

    report2 = scan_directory(d2)
    things = report2.tables["things"]
    assert "id" in things.columns
    assert "nested_id" in things.columns
    assert "label" in things.columns
    assert things.columns["id"].n_unique == 5
    assert things.columns["nested_id"].n_unique == 5


def _shipments_xml(n: int) -> str:
    root = ET.Element("shipments")
    for i in range(n):
        s = ET.SubElement(root, "shipment")
        ET.SubElement(s, "shipment_id").text = str(i)
        ET.SubElement(s, "warehouse").text = ["W1", "W2"][i % 2]
        routing = ET.SubElement(s, "routing")
        ET.SubElement(routing, "carrier").text = ["UPS", "DHL"][i % 2]
        ET.SubElement(s, "created_at").text = f"2022-02-{(i % 28) + 1:02d}T00:00:00"
        if i % 3 == 0:
            ET.SubElement(s, "notes").text = "handled with care"
    return ET.tostring(root, encoding="unicode")


def test_xml_loading(tmp_path):
    (tmp_path / "shipments.xml").write_text(_shipments_xml(15))

    report = scan_directory(tmp_path)
    ships = report.tables["shipments"]
    assert ships.rows == 15
    assert ships.pk == "shipment_id"
    assert ships.columns["shipment_id"].type == "int64"
    assert ships.columns["warehouse"].type == "string"
    assert ships.columns["carrier"].type == "string"
    assert ships.columns["created_at"].type == "timestamp"
    assert ships.columns["notes"].null_rate > 0


def test_xml_subdirectory_as_table(tmp_path):
    d = tmp_path / "shipments"
    d.mkdir()
    # Two files, sorted file order, streamed as one logical table.
    (d / "part0.xml").write_text(_shipments_xml(5))
    root = ET.Element("shipments")
    for i in range(5, 10):
        s = ET.SubElement(root, "shipment")
        ET.SubElement(s, "shipment_id").text = str(i)
        ET.SubElement(s, "warehouse").text = ["W1", "W2"][i % 2]
        ET.SubElement(s, "routing")
        ET.SubElement(s, "created_at").text = f"2022-02-{(i % 28) + 1:02d}T00:00:00"
    (d / "part1.xml").write_text(ET.tostring(root, encoding="unicode"))

    report = scan_directory(tmp_path)
    ships = report.tables["shipments"]
    assert ships.rows == 10
    assert ships.pk == "shipment_id"
    assert ships.columns["shipment_id"].type == "int64"
    assert ships.columns["warehouse"].type == "string"


def test_xml_scan_row_cap(tmp_path, monkeypatch):
    (tmp_path / "shipments.xml").write_text(_shipments_xml(200))
    monkeypatch.setenv("VERISYNTH_XML_SCAN_ROWS", "50")

    report = scan_directory(tmp_path)
    assert report.tables["shipments"].rows == 50


# --------------------------------------------------------------------------
# In-table document columns (§11): scan-side expansion.
# --------------------------------------------------------------------------


def test_json_payload_column_expansion(tmp_path):
    n = 30
    rows = []
    for i in range(n):
        payload = None
        if i % 5 != 0:  # some null payload rows
            payload = json.dumps(
                {
                    "event": "click" if i % 2 == 0 else "view",
                    "created_at": f"2023-01-{(i % 28) + 1:02d}T00:00:00",
                    "device": {
                        "os": "ios" if i % 2 == 0 else "android",
                        "utm": {"source": "ads"},
                    },
                }
            )
        rows.append({"event_id": i, "payload": payload})
    pl.DataFrame(rows).write_parquet(tmp_path / "events.parquet")

    report = scan_directory(tmp_path)
    events = report.tables["events"]

    assert events.pk == "event_id"
    assert "payload" not in events.columns

    assert events.columns["event"].type == "string"
    assert events.columns["created_at"].type == "timestamp"
    assert events.columns["os"].type == "string"
    assert events.columns["source"].type == "string"

    null_rows = sum(1 for i in range(n) if i % 5 == 0)
    expected_null_rate = round(null_rows / n, 6)
    assert events.columns["event"].null_rate == expected_null_rate
    assert events.columns["created_at"].null_rate == expected_null_rate
    assert events.columns["os"].null_rate == expected_null_rate
    assert events.columns["source"].null_rate == expected_null_rate


def test_xml_payload_column_expansion(tmp_path):
    n = 20
    payloads = []
    for i in range(n):
        rec = ET.Element("event")
        ET.SubElement(rec, "kind").text = "click" if i % 2 == 0 else "view"
        device = ET.SubElement(rec, "device")
        ET.SubElement(device, "os").text = "ios" if i % 2 == 0 else "android"
        if i % 3 == 0:  # optional element -> missing in most records
            ET.SubElement(device, "model").text = "X1"
        payloads.append(ET.tostring(rec, encoding="unicode"))

    pl.DataFrame(
        {"event_id": np.arange(n, dtype=np.int64), "payload": payloads}
    ).write_parquet(tmp_path / "events.parquet")

    report = scan_directory(tmp_path)
    events = report.tables["events"]

    assert events.pk == "event_id"
    assert "payload" not in events.columns
    assert events.columns["kind"].type == "string"
    assert events.columns["os"].type == "string"
    assert events.columns["model"].null_rate > 0


def test_plain_text_column_not_expanded(tmp_path):
    # Some values start with '{' but aren't (all) valid JSON objects -- must
    # be left alone, not treated as a document payload column.
    notes = [(f"{{not json {i}" if i % 3 == 0 else f"hello there {i}") for i in range(10)]
    pl.DataFrame(
        {"thing_id": np.arange(10, dtype=np.int64), "note": notes}
    ).write_parquet(tmp_path / "things.parquet")

    report = scan_directory(tmp_path)
    things = report.tables["things"]
    assert "note" in things.columns
    assert things.columns["note"].type == "string"
    assert things.columns["note"].n_unique == 10


def test_json_payload_collision_with_existing_column(tmp_path):
    n = 10
    payloads = [json.dumps({"thing_id": i, "value": i * 2}) for i in range(n)]
    pl.DataFrame(
        {"thing_id": np.arange(n, dtype=np.int64), "payload": payloads}
    ).write_parquet(tmp_path / "things.parquet")

    report = scan_directory(tmp_path)
    things = report.tables["things"]

    assert "payload" not in things.columns
    assert "thing_id" in things.columns  # original column untouched
    assert things.columns["thing_id"].unique
    assert "payload_thing_id" in things.columns  # colliding leaf, prefixed
    assert "value" in things.columns


def test_mixed_directory_loading(tmp_path):
    pl.DataFrame({"widget_id": np.arange(5, dtype=np.int64), "name": ["a", "b", "c", "d", "e"]}).write_parquet(
        tmp_path / "widgets.parquet"
    )
    (tmp_path / "gadgets.json").write_text(
        json.dumps([{"gadget_id": i, "label": "g"} for i in range(5)])
    )
    (tmp_path / "shipments.xml").write_text(_shipments_xml(5))

    report = scan_directory(tmp_path)
    assert set(report.tables) == {"widgets", "gadgets", "shipments"}
    assert report.tables["widgets"].pk == "widget_id"
    assert report.tables["gadgets"].pk == "gadget_id"
    assert report.tables["shipments"].pk == "shipment_id"


# --------------------------------------------------------------------------
# Nested entity extraction (read side of format.nest): JSON list-of-struct
# columns and repeated XML elements become separate child tables with the
# relation detected.
# --------------------------------------------------------------------------


def test_json_nested_entities_become_child_table(tmp_path):
    records = []
    for i in range(30):
        records.append(
            {
                "order_id": i,
                "status": "ok",
                "lines": [
                    {"item_id": i * 10 + j, "sku": "ab"[j % 2]} for j in range(i % 3)
                ],
            }
        )
    (tmp_path / "orders.json").write_text(json.dumps(records))

    report = scan_directory(tmp_path)
    assert set(report.tables) == {"orders", "lines"}
    orders = report.tables["orders"]
    assert "lines" not in orders.columns  # list column extracted, not profiled
    lines = report.tables["lines"]
    # Parent id injected + child fields, exploded one row per nested record.
    assert {"order_id", "item_id", "sku"} <= set(lines.columns)
    assert lines.rows == sum(i % 3 for i in range(30))
    assert lines.pk == "item_id"
    rel = [r for r in report.relations if r.child == "lines" and r.parent == "orders"]
    assert rel and rel[0].child_column == "order_id"


def test_xml_repeated_and_container_nested_entities(tmp_path):
    parts = ["<orders>"]
    for i in range(20):
        lines = "".join(
            f"<line><item_id>{i * 10 + j}</item_id><sku>s{j}</sku></line>"
            for j in range(1 + i % 2)
        )
        parts.append(
            f"<order><order_id>{i}</order_id><status>ok</status>"
            f"<lines>{lines}</lines>"
            f"<note><author>bot</author><text>t{i}</text></note>"
            f"</order>"
        )
    parts.append("</orders>")
    (tmp_path / "orders.xml").write_text("".join(parts))

    report = scan_directory(tmp_path)
    # <lines> is a container of repeated entity records -> child table;
    # <note> is a single nested struct-like element -> flattened into the
    # parent as scalar columns, NOT extracted.
    assert set(report.tables) == {"orders", "lines"}
    orders = report.tables["orders"]
    assert {"order_id", "status", "author", "text"} <= set(orders.columns)
    lines = report.tables["lines"]
    assert lines.rows == sum(1 + i % 2 for i in range(20))
    assert {"order_id", "item_id", "sku"} <= set(lines.columns)
    assert lines.columns["item_id"].type == "int64"
    rel = [r for r in report.relations if r.child == "lines" and r.parent == "orders"]
    assert rel and rel[0].child_column == "order_id"
