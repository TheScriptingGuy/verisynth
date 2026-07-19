"""Acceptance tests for verisynth.documents (json/jsonl/xml document rendering).

See docs/ARCHITECTURE.md §11 (normative).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime

import duckdb
import pytest

import verisynth.documents as documents_module
from verisynth.documents import (
    DocumentError,
    document_path,
    validate_documents,
    write_documents,
)
from verisynth.engine import Engine
from verisynth.metadata import FormatSpec, parse_metadata

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


# --------------------------------------------------------------------------
# 9. Streaming: large tables, batch-size-independent byte determinism, and
#    zero-row tables (see docs/ARCHITECTURE.md §11 streaming addendum).
# --------------------------------------------------------------------------

BIG_SEED = 7


def _big_metadata_dict(n: int, fmt: dict | None = None) -> dict:
    d: dict = {
        "version": 1,
        "seed": BIG_SEED,
        "tables": {
            "big": {
                "role": "root",
                "rows": n,
                "primary_key": "id",
                "columns": {
                    "id": {"type": "int64", "generator": "key"},
                    "val": {
                        "type": "string",
                        "distribution": {
                            "kind": "categorical",
                            "categories": ["a", "b", "c"],
                            "probs": [0.3, 0.3, 0.4],
                        },
                    },
                    "note": {
                        "type": "string",
                        "distribution": {
                            "kind": "categorical",
                            "categories": ["x", "y"],
                            "probs": [0.5, 0.5],
                        },
                        "null_rate": 0.4,
                    },
                },
            }
        },
    }
    if fmt is not None:
        d["tables"]["big"]["format"] = fmt
    return d


_BIG_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "val": {"type": "string"},
            "note": {"type": "string"},
        },
        "required": ["id", "val"],
    },
}

_BIG_XSD = """<?xml version="1.0"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <xs:element name="item">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="id" type="xs:long"/>
        <xs:element name="val" type="xs:string"/>
        <xs:element name="note" type="xs:string" minOccurs="0"/>
      </xs:sequence>
    </xs:complexType>
  </xs:element>
</xs:schema>
"""


def test_streaming_large_table_flat_xml_shaped_json_shaped_xml(tmp_path, monkeypatch):
    """~150k-row table, rendered with a small batch size (patched module
    constant): flat xml, schema-shaped json and schema-shaped xml must all
    stream to a valid, fully-accounted-for document."""
    monkeypatch.setattr(documents_module, "DEFAULT_FETCH_ROWS", 1000)

    (tmp_path / "big.schema.json").write_text(json.dumps(_BIG_JSON_SCHEMA))
    (tmp_path / "big.xsd").write_text(_BIG_XSD)

    n = 150_000
    md = parse_metadata(_big_metadata_dict(n=n, fmt={"kind": "xml"}))
    md.base_dir = tmp_path
    out_dir = tmp_path / "out"
    Engine(md, seed=BIG_SEED).generate(str(out_dir), num_partitions=2)

    # Flat xml.
    write_documents(md, out_dir)
    assert validate_documents(md, out_dir) == []
    p_xml = document_path(out_dir, md.tables["big"])
    root = ET.parse(p_xml).getroot()
    assert len(list(root)) == n

    # Schema-shaped json.
    md.tables["big"].format = FormatSpec(kind="json", schemas=["big.schema.json"])
    write_documents(md, out_dir)
    assert validate_documents(md, out_dir) == []
    p_json = document_path(out_dir, md.tables["big"])
    with open(p_json) as f:
        records = json.load(f)
    assert len(records) == n

    # Schema-shaped xml.
    md.tables["big"].format = FormatSpec(kind="xml", schemas=["big.xsd"])
    write_documents(md, out_dir)
    assert validate_documents(md, out_dir) == []
    p_xml2 = document_path(out_dir, md.tables["big"])
    root2 = ET.parse(p_xml2).getroot()
    assert len(list(root2)) == n
    assert list(root2)[0].tag == "item"


def test_write_documents_byte_identical_across_batch_sizes(tmp_path, monkeypatch):
    """Schema-shaped json rendered with a tiny batch size must be byte-for-byte
    identical to the same table rendered with one huge batch."""
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "val": {"type": "string"}},
            "required": ["id", "val"],
        },
    }
    (tmp_path / "s.schema.json").write_text(json.dumps(schema))
    fmt = {"kind": "json", "schemas": ["s.schema.json"]}

    n = 5000
    md = parse_metadata(_big_metadata_dict(n=n, fmt=fmt))
    md.base_dir = tmp_path
    out_dir = tmp_path / "out"
    Engine(md, seed=BIG_SEED).generate(str(out_dir), num_partitions=2)
    p = document_path(out_dir, md.tables["big"])

    monkeypatch.setattr(documents_module, "DEFAULT_FETCH_ROWS", 7)
    write_documents(md, out_dir)
    small_batch_bytes = p.read_bytes()

    monkeypatch.setattr(documents_module, "DEFAULT_FETCH_ROWS", 10_000_000)
    write_documents(md, out_dir)
    huge_batch_bytes = p.read_bytes()

    assert small_batch_bytes == huge_batch_bytes
    assert json.loads(small_batch_bytes) is not None


def test_empty_table_shaped_json_document(tmp_path):
    """A zero-row child table (root has 1 row; bernoulli p is tiny enough
    that no children are produced) must still compile its json schema and
    render a valid, empty document."""
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "child_id": {"type": "integer"},
                "val": {"type": "string"},
            },
            "required": ["child_id"],
        },
    }
    (tmp_path / "child.schema.json").write_text(json.dumps(schema))
    d = {
        "version": 1,
        "seed": SEED,
        "tables": {
            "parent": {
                "role": "root",
                "rows": 1,
                "primary_key": "parent_id",
                "columns": {
                    "parent_id": {"type": "int64", "generator": "key"},
                },
            },
            "child": {
                "role": "child",
                "parent": "parent",
                "cardinality": {"kind": "bernoulli", "p": 0.000001},
                "child_stride": 16,
                "primary_key": "child_id",
                "columns": {
                    "child_id": {"type": "int64", "generator": "key"},
                    "parent_id": {"type": "int64", "generator": "parent_key"},
                    "val": {
                        "type": "string",
                        "distribution": {
                            "kind": "categorical",
                            "categories": ["a", "b"],
                            "probs": [0.5, 0.5],
                        },
                    },
                },
                "format": {"kind": "json", "schemas": ["child.schema.json"]},
            },
        },
    }
    md = parse_metadata(d)
    md.base_dir = tmp_path
    out_dir = tmp_path / "out"
    Engine(md, seed=SEED).generate(str(out_dir), num_partitions=1)

    glob = str(out_dir / "child" / "*.parquet")
    con = duckdb.connect()
    try:
        (n_child,) = con.execute(f"SELECT count(*) FROM read_parquet('{glob}')").fetchone()
    finally:
        con.close()
    assert n_child == 0

    write_documents(md, out_dir)
    p = document_path(out_dir, md.tables["child"])
    assert p.read_text() == "[]\n"
    with open(p) as f:
        records = json.load(f)
    assert records == []

    assert validate_documents(md, out_dir) == []


# --------------------------------------------------------------------------
# 10. In-table document columns (docs/ARCHITECTURE.md §11): a column whose
# per-row value is a JSON object / XML fragment rendered from the row's own
# sibling columns.
# --------------------------------------------------------------------------


def _doc_column_metadata(payload_spec: dict, extra_columns: dict | None = None, rows: int = 80):
    d = {
        "version": 1,
        "seed": SEED,
        "tables": {
            "events": {
                "role": "root",
                "rows": rows,
                "primary_key": "event_id",
                "columns": {
                    "event_id": {"type": "int64", "generator": "key"},
                    "amount": {
                        "type": "float64",
                        "distribution": {"kind": "uniform", "low": 1.0, "high": 9.0},
                        "null_rate": 0.3,
                    },
                    "device": {
                        "type": "string",
                        "distribution": {
                            "kind": "categorical",
                            "categories": ["mobile", "desktop"],
                            "probs": [0.6, 0.4],
                        },
                    },
                    "flag": {
                        "type": "bool",
                        "distribution": {
                            "kind": "categorical",
                            "categories": [False, True],
                            "probs": [0.5, 0.5],
                        },
                    },
                    "at": {
                        "type": "timestamp",
                        "distribution": {
                            "kind": "datetime_uniform",
                            "start": "2024-01-01T00:00:00",
                            "end": "2024-06-01T00:00:00",
                        },
                    },
                    "payload": {"type": "string", "document": payload_spec},
                    **(extra_columns or {}),
                },
            }
        },
    }
    return parse_metadata(d)


def _generated_events(md, num_partitions: int = 2):
    tables = []
    for p in range(num_partitions):
        tables.append(Engine(md, seed=SEED).generate_partition(p, num_partitions)["events"])
    import pyarrow as pa

    return pa.concat_tables(tables).to_pylist()


def test_document_column_flat_json_agrees_with_row():
    md = _doc_column_metadata({"kind": "json"})
    rows = _generated_events(md)
    assert rows
    saw_null_amount = False
    for row in rows:
        cell = row["payload"]
        assert cell is not None and "\n" not in cell
        payload = json.loads(cell)
        # Default embedding: every non-document sibling, declaration order.
        assert list(payload) == ["event_id", "amount", "device", "flag", "at"]
        assert payload["event_id"] == row["event_id"]
        assert payload["device"] == row["device"]
        assert payload["flag"] == row["flag"]
        assert payload["amount"] == row["amount"]  # None -> JSON null -> None
        if row["amount"] is None:
            saw_null_amount = True
        assert datetime.fromisoformat(payload["at"]) == row["at"]
    assert saw_null_amount


def test_document_column_explicit_subset_and_mutual_exclusion():
    md = _doc_column_metadata(
        {"kind": "json", "columns": ["device", "flag"]},
        extra_columns={
            "payload2": {"type": "string", "document": {"kind": "json"}},
        },
    )
    rows = _generated_events(md, num_partitions=1)
    for row in rows:
        assert list(json.loads(row["payload"])) == ["device", "flag"]
        # payload2 uses the default embedding, which excludes BOTH document
        # columns -- never nests one payload inside another.
        assert list(json.loads(row["payload2"])) == [
            "event_id",
            "amount",
            "device",
            "flag",
            "at",
        ]


def test_document_column_flat_xml_root_and_null_omission():
    md = _doc_column_metadata({"kind": "xml"})
    rows = _generated_events(md)
    for row in rows:
        cell = row["payload"]
        assert "\n" not in cell
        el = ET.fromstring(cell)
        assert el.tag == "event"  # _singular("events")
        assert (el.find("amount") is None) == (row["amount"] is None)
        assert el.find("device").text == row["device"]

    md2 = _doc_column_metadata({"kind": "xml", "root": "evt"})
    rows2 = _generated_events(md2, num_partitions=1)
    assert ET.fromstring(rows2[0]["payload"]).tag == "evt"


def test_document_column_schema_shaped_json(tmp_path):
    common = {
        "$id": "doccol_common.schema.json",
        "definitions": {
            "meta": {
                "type": "object",
                "properties": {
                    "device": {"type": "string"},
                    "flag": {"type": "boolean"},
                },
            }
        },
    }
    primary = {
        "type": "object",
        "required": ["event_id"],
        "properties": {
            "event_id": {"type": "integer"},
            "meta": {"$ref": "doccol_common.schema.json#/definitions/meta"},
        },
    }
    (tmp_path / "doccol_common.schema.json").write_text(json.dumps(common))
    (tmp_path / "doccol.schema.json").write_text(json.dumps(primary))

    md = _doc_column_metadata(
        {"kind": "json", "schemas": ["doccol.schema.json", "doccol_common.schema.json"]}
    )
    md.base_dir = tmp_path
    rows = _generated_events(md, num_partitions=1)
    for row in rows:
        payload = json.loads(row["payload"])
        assert payload["event_id"] == row["event_id"]
        assert payload["meta"] == {"device": row["device"], "flag": row["flag"]}


def test_document_column_schema_missing_required_raises(tmp_path):
    (tmp_path / "bad.schema.json").write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["nope"],
                "properties": {"nope": {"type": "string"}},
            }
        )
    )
    md = _doc_column_metadata({"kind": "json", "schemas": ["bad.schema.json"]})
    md.base_dir = tmp_path
    with pytest.raises(DocumentError, match="nope"):
        Engine(md, seed=SEED).generate_partition(0, 1)


def test_document_column_xsd_shaped(tmp_path):
    (tmp_path / "doccol_common.xsd").write_text(
        """<?xml version="1.0"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <xs:complexType name="MetaType">
    <xs:sequence>
      <xs:element name="device" type="xs:string"/>
      <xs:element name="flag" type="xs:boolean"/>
    </xs:sequence>
  </xs:complexType>
</xs:schema>"""
    )
    (tmp_path / "doccol.xsd").write_text(
        """<?xml version="1.0"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <xs:include schemaLocation="doccol_common.xsd"/>
  <xs:element name="event">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="event_id" type="xs:long"/>
        <xs:element name="meta" type="MetaType"/>
        <xs:element name="amount" type="xs:double" minOccurs="0"/>
      </xs:sequence>
    </xs:complexType>
  </xs:element>
</xs:schema>"""
    )
    md = _doc_column_metadata(
        {"kind": "xml", "root": "event", "schemas": ["doccol.xsd", "doccol_common.xsd"]}
    )
    md.base_dir = tmp_path
    rows = _generated_events(md, num_partitions=1)
    for row in rows:
        el = ET.fromstring(row["payload"])
        expected = ["event_id", "meta"] + ([] if row["amount"] is None else ["amount"])
        assert [c.tag for c in el] == expected
        assert [c.tag for c in el.find("meta")] == ["device", "flag"]
        assert int(el.find("event_id").text) == row["event_id"]


def test_document_column_null_rate():
    md = _doc_column_metadata({"kind": "json"}, rows=400)
    md.tables["events"].columns["payload"].null_rate = 0.3
    rows = _generated_events(md)
    frac = sum(1 for r in rows if r["payload"] is None) / len(rows)
    assert abs(frac - 0.3) < 0.1
    assert any(r["payload"] is not None for r in rows)


def test_document_column_partition_consistency():
    import pyarrow as pa

    md = _doc_column_metadata({"kind": "json"})
    single = Engine(md, seed=SEED).generate_partition(0, 1)["events"]
    parts = [Engine(md, seed=SEED).generate_partition(p, 3)["events"] for p in range(3)]
    assert pa.concat_tables(parts).equals(single)


# --------------------------------------------------------------------------
# 11. Relational nesting (format.nest): child-table rows as nested arrays /
# repeated elements inside the parent's document.
# --------------------------------------------------------------------------


def _nested_metadata(fmt: dict) -> "object":
    d = _base_metadata_dict(n_orders=40)
    # Grandchild of items, to exercise recursive nesting.
    d["tables"]["item_notes"] = {
        "role": "child",
        "parent": "items",
        "cardinality": {"kind": "poisson", "lam": 0.8, "max": 4},
        "child_stride": 8,
        "primary_key": "note_id",
        "columns": {
            "note_id": {"type": "int64", "generator": "key"},
            "item_id": {"type": "int64", "generator": "parent_key"},
            "text": {
                "type": "string",
                "distribution": {
                    "kind": "categorical",
                    "categories": ["ok", "late", "damaged"],
                    "probs": [0.6, 0.25, 0.15],
                },
            },
        },
    }
    d["tables"]["orders"]["format"] = fmt
    return parse_metadata(d)


def _expected_children(out_dir, child: str, fk: str, cols: str, pk: str) -> dict:
    glob = str(out_dir / child / "*.parquet")
    grouped: dict = {}
    for row in _duckdb_rows(glob, f"{fk}, {cols}", f"{fk}, {pk}"):
        grouped.setdefault(row[0], []).append(row[1:])
    return grouped


def test_nested_flat_json(tmp_path):
    md = _nested_metadata(
        {
            "kind": "json",
            "nest": [
                {"table": "items", "as": "lines", "nest": [{"table": "item_notes", "as": "notes"}]}
            ],
        }
    )
    out_dir = tmp_path / "out"
    Engine(md, seed=SEED).generate(str(out_dir), num_partitions=2)

    with open(document_path(out_dir, md.tables["orders"])) as f:
        records = json.load(f)
    assert len(records) == 40

    by_order = _expected_children(out_dir, "items", "order_id", "item_id, sku", "item_id")
    by_item = _expected_children(out_dir, "item_notes", "item_id", "note_id, text", "note_id")

    total_items = 0
    saw_childless = False
    for rec in records:
        lines = rec["lines"]
        expected = by_order.get(rec["order_id"], [])
        # fk column excluded by default; child pk order preserved.
        assert [(ln["item_id"], ln["sku"]) for ln in lines] == expected
        if not expected:
            saw_childless = True
            assert lines == []
        total_items += len(lines)
        for ln in lines:
            assert [(nt["note_id"], nt["text"]) for nt in ln["notes"]] == by_item.get(
                ln["item_id"], []
            )
            assert "order_id" not in ln  # parent_key link is redundant inside the parent
    assert saw_childless
    assert total_items == sum(len(v) for v in by_order.values())

    assert validate_documents(md, out_dir) == []


def test_nested_shaped_json_array_property(tmp_path):
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["order_id", "lines"],
            "properties": {
                "order_id": {"type": "integer"},
                "customer": {
                    "type": "object",
                    "properties": {"customer_name": {"type": "string"}},
                },
                "lines": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["sku"],
                        "properties": {
                            "item_id": {"type": "integer"},
                            "sku": {"type": "string"},
                        },
                    },
                },
            },
        },
    }
    (tmp_path / "nested.schema.json").write_text(json.dumps(schema))
    md = _nested_metadata(
        {
            "kind": "json",
            "schemas": ["nested.schema.json"],
            "nest": [{"table": "items", "as": "lines"}],
        }
    )
    md.base_dir = tmp_path
    out_dir = tmp_path / "out"
    Engine(md, seed=SEED).generate(str(out_dir), num_partitions=2)

    with open(document_path(out_dir, md.tables["orders"])) as f:
        records = json.load(f)

    by_order = _expected_children(out_dir, "items", "order_id", "item_id, sku", "item_id")
    for rec in records:
        assert set(rec) <= {"order_id", "customer", "lines"}
        assert [(ln["item_id"], ln["sku"]) for ln in rec["lines"]] == by_order.get(
            rec["order_id"], []
        )


def test_nested_shaped_json_array_without_nest_raises(tmp_path):
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["lines"],
            "properties": {"lines": {"type": "array", "items": {"type": "object"}}},
        },
    }
    (tmp_path / "bad.schema.json").write_text(json.dumps(schema))
    md = _nested_metadata({"kind": "json", "schemas": ["bad.schema.json"]})
    md.base_dir = tmp_path
    with pytest.raises(DocumentError, match="no matching nested child"):
        Engine(md, seed=SEED).generate(str(tmp_path / "out"), num_partitions=1)


def test_nested_flat_xml(tmp_path):
    md = _nested_metadata({"kind": "xml", "nest": [{"table": "items", "as": "lines"}]})
    out_dir = tmp_path / "out"
    Engine(md, seed=SEED).generate(str(out_dir), num_partitions=2)

    root = ET.parse(document_path(out_dir, md.tables["orders"])).getroot()
    by_order = _expected_children(out_dir, "items", "order_id", "item_id, sku", "item_id")
    for rec in root:
        order_id = int(rec.find("order_id").text)
        container = rec.find("lines")
        assert container is not None
        assert all(item.tag == "line" for item in container)  # singular(alias)
        got = [(int(i.find("item_id").text), i.find("sku").text) for i in container]
        assert got == by_order.get(order_id, [])


def test_nested_xsd_shaped_xml(tmp_path):
    (tmp_path / "nested.xsd").write_text(
        """<?xml version="1.0"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <xs:element name="orders">
    <xs:complexType><xs:sequence>
      <xs:element ref="order" minOccurs="0" maxOccurs="unbounded"/>
    </xs:sequence></xs:complexType>
  </xs:element>
  <xs:element name="order">
    <xs:complexType><xs:sequence>
      <xs:element name="order_id" type="xs:long"/>
      <xs:element name="lines">
        <xs:complexType><xs:sequence>
          <xs:element name="line" maxOccurs="unbounded" minOccurs="0">
            <xs:complexType><xs:sequence>
              <xs:element name="item_id" type="xs:long"/>
              <xs:element name="sku" type="xs:string"/>
            </xs:sequence></xs:complexType>
          </xs:element>
        </xs:sequence></xs:complexType>
      </xs:element>
    </xs:sequence></xs:complexType>
  </xs:element>
</xs:schema>"""
    )
    md = _nested_metadata(
        {
            "kind": "xml",
            "root": "orders",
            "record": "order",
            "schemas": ["nested.xsd"],
            # The XSD wraps repeated <line> in <lines>; bind the container
            # name.
            "nest": [{"table": "items", "as": "lines"}],
        }
    )
    md.base_dir = tmp_path
    out_dir = tmp_path / "out"
    Engine(md, seed=SEED).generate(str(out_dir), num_partitions=2)

    root = ET.parse(document_path(out_dir, md.tables["orders"])).getroot()
    by_order = _expected_children(out_dir, "items", "order_id", "item_id, sku", "item_id")
    assert len(list(root)) == 40
    for rec in root:
        assert [c.tag for c in rec] == ["order_id", "lines"]
        got = [
            (int(i.find("item_id").text), i.find("sku").text) for i in rec.find("lines")
        ]
        assert got == by_order.get(int(rec.find("order_id").text), [])


def test_nested_validation_catches_truncated_children(tmp_path):
    md = _nested_metadata({"kind": "json", "nest": [{"table": "items"}]})
    out_dir = tmp_path / "out"
    Engine(md, seed=SEED).generate(str(out_dir), num_partitions=1)
    assert validate_documents(md, out_dir) == []

    p = document_path(out_dir, md.tables["orders"])
    records = json.loads(p.read_text())
    for rec in records:
        rec["items"] = rec["items"][:1]
    p.write_text(json.dumps(records))
    assert any("nests" in v for v in validate_documents(md, out_dir))


def test_nested_document_determinism(tmp_path):
    md = _nested_metadata({"kind": "json", "nest": [{"table": "items", "as": "lines"}]})
    out1, out2 = tmp_path / "a", tmp_path / "b"
    Engine(md, seed=SEED).generate(str(out1), num_partitions=1)
    Engine(md, seed=SEED).generate(str(out2), num_partitions=3)
    p1 = document_path(out1, md.tables["orders"])
    p2 = document_path(out2, md.tables["orders"])
    assert p1.read_bytes() == p2.read_bytes()
