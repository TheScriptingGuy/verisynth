"""Acceptance tests for verisynth.documents (json/jsonl/xml document rendering).

See docs/ARCHITECTURE.md §11 (normative).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime

import duckdb
import pytest

from verisynth.documents import (
    DocumentError,
    document_path,
    validate_documents,
    write_documents,
)
from verisynth.engine import Engine
from verisynth.metadata import parse_metadata

SEED = 42


# --------------------------------------------------------------------------
# Base metadata: root "orders" (int64/float64/string/bool/timestamp, one
# nullable column) + child "items".
# --------------------------------------------------------------------------


def _base_metadata_dict(n_orders: int = 60) -> dict:
    return {
        "version": 1,
        "seed": SEED,
        "tables": {
            "orders": {
                "role": "root",
                "rows": n_orders,
                "primary_key": "order_id",
                "columns": {
                    "order_id": {"type": "int64", "generator": "key"},
                    "amount": {
                        "type": "float64",
                        "distribution": {"kind": "uniform", "low": 10.0, "high": 500.0},
                        "null_rate": 0.3,
                    },
                    "customer_name": {
                        "type": "string",
                        "distribution": {
                            "kind": "categorical",
                            "categories": ["alice", "bob", "carol"],
                            "probs": [0.4, 0.3, 0.3],
                        },
                    },
                    "is_paid": {
                        "type": "bool",
                        "distribution": {
                            "kind": "categorical",
                            "categories": [False, True],
                            "probs": [0.5, 0.5],
                        },
                    },
                    "placed_at": {
                        "type": "timestamp",
                        "distribution": {
                            "kind": "datetime_uniform",
                            "start": "2022-01-01T00:00:00",
                            "end": "2023-01-01T00:00:00",
                        },
                    },
                },
            },
            "items": {
                "role": "child",
                "parent": "orders",
                "cardinality": {"kind": "poisson", "lam": 2.0, "max": 10},
                "child_stride": 16,
                "primary_key": "item_id",
                "columns": {
                    "item_id": {"type": "int64", "generator": "key"},
                    "order_id": {"type": "int64", "generator": "parent_key"},
                    "sku": {
                        "type": "string",
                        "distribution": {
                            "kind": "categorical",
                            "categories": ["a", "b", "c"],
                            "probs": [0.3, 0.3, 0.4],
                        },
                    },
                },
            },
        },
    }


def _metadata_with_format(fmt: dict, n_orders: int = 60):
    d = _base_metadata_dict(n_orders=n_orders)
    d["tables"]["orders"]["format"] = fmt
    return parse_metadata(d)


def _duckdb_rows(glob: str, select: str, order_by: str) -> list[tuple]:
    con = duckdb.connect()
    try:
        return con.execute(
            f"SELECT {select} FROM read_parquet('{glob}') ORDER BY {order_by}"
        ).fetchall()
    finally:
        con.close()


# --------------------------------------------------------------------------
# 1. Flat json array
# --------------------------------------------------------------------------


def test_flat_json_array(tmp_path):
    md = _metadata_with_format({"kind": "json"})
    eng = Engine(md, seed=SEED)
    out_dir = tmp_path / "out"
    eng.generate(str(out_dir), num_partitions=2)

    paths = write_documents(md, out_dir)
    p = document_path(out_dir, md.tables["orders"])
    assert paths["orders"] == p
    assert p.exists()
    assert p == out_dir / "orders.json"

    with open(p) as f:
        records = json.load(f)

    glob = str(out_dir / "orders" / "*.parquet")
    rows = _duckdb_rows(glob, "*", "order_id")

    assert len(records) == len(rows) == md.tables["orders"].rows

    cols = ["order_id", "amount", "customer_name", "is_paid", "placed_at"]
    assert list(records[0].keys()) == cols

    order_ids = [r["order_id"] for r in records]
    expected_ids = [row[cols.index("order_id")] for row in rows]
    assert order_ids == expected_ids

    names = [r["customer_name"] for r in records]
    expected_names = [row[cols.index("customer_name")] for row in rows]
    assert names == expected_names

    assert any(r["amount"] is None for r in records)


# --------------------------------------------------------------------------
# 2. jsonl
# --------------------------------------------------------------------------


def test_jsonl(tmp_path):
    md = _metadata_with_format({"kind": "jsonl"})
    eng = Engine(md, seed=SEED)
    out_dir = tmp_path / "out"
    eng.generate(str(out_dir), num_partitions=2)
    write_documents(md, out_dir)

    p = document_path(out_dir, md.tables["orders"])
    assert p == out_dir / "orders.jsonl"

    lines = p.read_text().splitlines()
    assert len(lines) == md.tables["orders"].rows

    records = [json.loads(line) for line in lines]
    assert len(records) == len(lines)
    assert all("order_id" in r for r in records)


# --------------------------------------------------------------------------
# 3. Flat xml
# --------------------------------------------------------------------------


def test_flat_xml_default_names_and_null_omission(tmp_path):
    md = _metadata_with_format({"kind": "xml"})
    eng = Engine(md, seed=SEED)
    out_dir = tmp_path / "out"
    eng.generate(str(out_dir), num_partitions=2)
    write_documents(md, out_dir)

    p = document_path(out_dir, md.tables["orders"])
    assert p == out_dir / "orders.xml"

    root = ET.parse(p).getroot()
    assert root.tag == "orders"
    records = list(root)
    assert records
    assert records[0].tag == "order"

    glob = str(out_dir / "orders" / "*.parquet")
    rows = _duckdb_rows(glob, "order_id, amount", "order_id")
    null_order_ids = {r[0] for r in rows if r[1] is None}
    assert null_order_ids

    for rec in records:
        order_id = int(rec.find("order_id").text)
        has_amount = rec.find("amount") is not None
        if order_id in null_order_ids:
            assert not has_amount
        else:
            assert has_amount


def test_flat_xml_custom_root_and_record_names(tmp_path):
    md = _metadata_with_format({"kind": "xml", "root": "OrderList", "record": "OrderRec"})
    eng = Engine(md, seed=SEED)
    out_dir = tmp_path / "out"
    eng.generate(str(out_dir), num_partitions=2)
    write_documents(md, out_dir)

    p = document_path(out_dir, md.tables["orders"])
    root = ET.parse(p).getroot()
    assert root.tag == "OrderList"
    assert list(root)[0].tag == "OrderRec"


# --------------------------------------------------------------------------
# 4/5. Schema-shaped json (nested object + $ref, and compile-time errors).
# --------------------------------------------------------------------------

EVENTS_SEED = 3


def _events_metadata_dict(fmt: dict | None = None, n: int = 200) -> dict:
    d = {
        "version": 1,
        "seed": EVENTS_SEED,
        "tables": {
            "events": {
                "role": "root",
                "rows": n,
                "primary_key": "id",
                "columns": {
                    "id": {"type": "int64", "generator": "key"},
                    "placed_at": {
                        "type": "timestamp",
                        "distribution": {
                            "kind": "datetime_uniform",
                            "start": "2022-01-01T00:00:00",
                            "end": "2023-01-01T00:00:00",
                        },
                        "null_rate": 0.2,
                    },
                    "device": {
                        "type": "string",
                        "distribution": {
                            "kind": "categorical",
                            "categories": ["mobile", "desktop"],
                            "probs": [0.5, 0.5],
                        },
                    },
                    "utm_source": {
                        "type": "string",
                        "distribution": {
                            "kind": "categorical",
                            "categories": ["google", "direct"],
                            "probs": [0.6, 0.4],
                        },
                        "null_rate": 0.3,
                    },
                },
            }
        },
    }
    if fmt is not None:
        d["tables"]["events"]["format"] = fmt
    return d


def test_schema_shaped_json_nested_and_ref(tmp_path):
    common_schema = {
        "definitions": {
            "attribution": {
                "type": "object",
                "properties": {
                    "device": {"type": "string"},
                    "utm_source": {"type": "string"},
                },
                "required": ["device"],
            }
        }
    }
    primary_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "placed_at": {"type": "string"},
                "attribution": {"$ref": "common.schema.json#/definitions/attribution"},
            },
            "required": ["id", "placed_at"],
        },
    }
    (tmp_path / "events.schema.json").write_text(json.dumps(primary_schema))
    (tmp_path / "common.schema.json").write_text(json.dumps(common_schema))

    fmt = {"kind": "json", "schemas": ["events.schema.json", "common.schema.json"]}
    md = parse_metadata(_events_metadata_dict(fmt=fmt, n=200))
    md.base_dir = tmp_path

    out_dir = tmp_path / "out"
    Engine(md, seed=EVENTS_SEED).generate(str(out_dir), num_partitions=2)
    write_documents(md, out_dir)

    p = document_path(out_dir, md.tables["events"])
    records = json.loads(p.read_text())

    glob = str(out_dir / "events" / "*.parquet")
    rows = _duckdb_rows(glob, "id, placed_at, device, utm_source", "id")

    assert len(records) == len(rows)

    saw_missing_utm = False
    for rec, (rid, placed_at, device, utm_source) in zip(records, rows):
        assert rec["id"] == rid
        assert isinstance(rec["id"], int)
        assert set(rec.keys()) == {"id", "placed_at", "attribution"}

        if placed_at is None:
            assert rec["placed_at"] is None
        else:
            assert datetime.fromisoformat(rec["placed_at"]) == placed_at

        assert rec["attribution"]["device"] == device
        if utm_source is None:
            assert "utm_source" not in rec["attribution"]
            saw_missing_utm = True
        else:
            assert rec["attribution"]["utm_source"] == utm_source

    assert saw_missing_utm


def test_schema_shaped_json_missing_required_column_raises(tmp_path):
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "nonexistent": {"type": "string"},
            },
            "required": ["id", "nonexistent"],
        },
    }
    (tmp_path / "bad.schema.json").write_text(json.dumps(schema))
    fmt = {"kind": "json", "schemas": ["bad.schema.json"]}
    md = parse_metadata(_events_metadata_dict(fmt=fmt, n=10))
    md.base_dir = tmp_path

    out_dir = tmp_path / "out"
    # Engine.generate itself renders documents after the parquet partitions,
    # so the compile-time schema check surfaces there already.
    with pytest.raises(DocumentError, match="nonexistent"):
        Engine(md, seed=EVENTS_SEED).generate(str(out_dir), num_partitions=1)

    with pytest.raises(DocumentError, match="nonexistent"):
        write_documents(md, out_dir)


# --------------------------------------------------------------------------
# 6. XSD-shaped xml (xs:include, unbounded wrapper particle, nesting, minOccurs=0)
# --------------------------------------------------------------------------

SHIP_SEED = 5

COMMON_XSD = """<?xml version="1.0"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <xs:complexType name="DestinationType">
    <xs:sequence>
      <xs:element name="city" type="xs:string"/>
      <xs:element name="zip" type="xs:string" minOccurs="0"/>
    </xs:sequence>
  </xs:complexType>
</xs:schema>
"""

SHIPMENTS_XSD = """<?xml version="1.0"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <xs:include schemaLocation="common.xsd"/>
  <xs:element name="shipments">
    <xs:complexType>
      <xs:sequence>
        <xs:element ref="shipment" maxOccurs="unbounded"/>
      </xs:sequence>
    </xs:complexType>
  </xs:element>
  <xs:element name="shipment">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="ship_id" type="xs:long"/>
        <xs:element name="status" type="xs:string"/>
        <xs:element name="weight" type="xs:decimal"/>
        <xs:element name="destination" type="DestinationType"/>
      </xs:sequence>
    </xs:complexType>
  </xs:element>
</xs:schema>
"""


def _shipments_metadata_dict(fmt: dict | None = None, n: int = 150) -> dict:
    d = {
        "version": 1,
        "seed": SHIP_SEED,
        "tables": {
            "shipments": {
                "role": "root",
                "rows": n,
                "primary_key": "ship_id",
                "columns": {
                    "ship_id": {"type": "int64", "generator": "key"},
                    "status": {
                        "type": "string",
                        "distribution": {
                            "kind": "categorical",
                            "categories": ["pending", "delivered"],
                            "probs": [0.4, 0.6],
                        },
                    },
                    "weight": {
                        "type": "float64",
                        "distribution": {"kind": "uniform", "low": 0.1, "high": 20.0},
                    },
                    "city": {
                        "type": "string",
                        "distribution": {
                            "kind": "categorical",
                            "categories": ["NYC", "LA"],
                            "probs": [0.5, 0.5],
                        },
                    },
                    "zip": {
                        "type": "string",
                        "distribution": {
                            "kind": "categorical",
                            "categories": ["10001", "90001"],
                            "probs": [0.5, 0.5],
                        },
                        "null_rate": 0.3,
                    },
                },
            }
        },
    }
    if fmt is not None:
        d["tables"]["shipments"]["format"] = fmt
    return d


def test_xsd_shaped_xml_nesting_order_and_include(tmp_path):
    (tmp_path / "common.xsd").write_text(COMMON_XSD)
    (tmp_path / "shipments.xsd").write_text(SHIPMENTS_XSD)

    fmt = {"kind": "xml", "schemas": ["shipments.xsd"]}
    md = parse_metadata(_shipments_metadata_dict(fmt=fmt, n=150))
    md.base_dir = tmp_path

    out_dir = tmp_path / "out"
    Engine(md, seed=SHIP_SEED).generate(str(out_dir), num_partitions=2)
    write_documents(md, out_dir)

    p = document_path(out_dir, md.tables["shipments"])
    root = ET.parse(p).getroot()
    assert root.tag == "shipments"
    records = list(root)
    assert records
    assert records[0].tag == "shipment"

    glob = str(out_dir / "shipments" / "*.parquet")
    rows = _duckdb_rows(glob, "ship_id, zip", "ship_id")
    null_zip_ids = {r[0] for r in rows if r[1] is None}
    assert null_zip_ids

    for rec in records:
        tags = [child.tag for child in rec]
        assert tags == ["ship_id", "status", "weight", "destination"]

        dest = rec.find("destination")
        dest_tags = [c.tag for c in dest]
        ship_id = int(rec.find("ship_id").text)
        if ship_id in null_zip_ids:
            assert dest_tags == ["city"]
        else:
            assert dest_tags == ["city", "zip"]


# --------------------------------------------------------------------------
# 7. validate_documents
# --------------------------------------------------------------------------


def test_validate_documents_ok_missing_and_truncated(tmp_path):
    md = _metadata_with_format({"kind": "json"})
    eng = Engine(md, seed=SEED)
    out_dir = tmp_path / "out"
    eng.generate(str(out_dir), num_partitions=2)
    write_documents(md, out_dir)

    assert validate_documents(md, out_dir) == []

    p = document_path(out_dir, md.tables["orders"])
    p.unlink()
    violations = validate_documents(md, out_dir)
    assert len(violations) == 1
    assert "missing" in violations[0]
    assert "orders" in violations[0]

    write_documents(md, out_dir)
    records = json.loads(p.read_text())
    p.write_text(json.dumps(records[:-1]))
    violations = validate_documents(md, out_dir)
    assert len(violations) == 1
    assert "has" in violations[0] and "expected" in violations[0]


# --------------------------------------------------------------------------
# 8. Determinism / partition invariance
# --------------------------------------------------------------------------


def test_write_documents_determinism_and_partition_invariance(tmp_path):
    md = _metadata_with_format({"kind": "json"}, n_orders=80)
    out_dir = tmp_path / "out"
    Engine(md, seed=SEED).generate(str(out_dir), num_partitions=2)
    write_documents(md, out_dir)
    p = document_path(out_dir, md.tables["orders"])
    content1 = p.read_bytes()

    write_documents(md, out_dir)
    content2 = p.read_bytes()
    assert content1 == content2

    md1 = _metadata_with_format({"kind": "json"}, n_orders=80)
    out_dir1 = tmp_path / "out1"
    Engine(md1, seed=SEED).generate(str(out_dir1), num_partitions=1)
    write_documents(md1, out_dir1)
    content_p1 = document_path(out_dir1, md1.tables["orders"]).read_bytes()

    md3 = _metadata_with_format({"kind": "json"}, n_orders=80)
    out_dir3 = tmp_path / "out3"
    Engine(md3, seed=SEED).generate(str(out_dir3), num_partitions=3)
    write_documents(md3, out_dir3)
    content_p3 = document_path(out_dir3, md3.tables["orders"]).read_bytes()

    assert content_p1 == content_p3
