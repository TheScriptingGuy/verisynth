"""Acceptance tests for verisynth.xmlstream: streaming XML reading (bounded
memory), Parquet staging, and parallel 100k-file batch ingestion.

See docs/ARCHITECTURE.md §12 and the xmlstream.py module docstring
(normative record semantics: records are direct children of the document
root, nested containers flatten like JSON structs with the same
``{owner_tag}_{leaf}`` collision fallback as scanner.py's JSON path, tag
names are reduced to local names, batches are all-string DataFrames).

No rust ``verisynth_kernels`` wheel with XML support is installed in this
environment, so ``xmlstream.BACKEND`` is already "reference"; the
reference-specific test below still monkeypatches ``_RUST`` to ``None`` per
the module's documented force-fallback contract so it stays correct even if
a rust wheel is later installed.
"""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import duckdb
import polars as pl
import pytest

import verisynth.xmlstream as xmlstream
from verisynth.xmlstream import (
    count_xml_records,
    iter_xml_batches,
    xml_dir_to_parquet,
    xml_to_parquet,
)


def _write_shipments(path: Path, n: int, start: int = 0) -> None:
    root = ET.Element("shipments")
    for i in range(start, start + n):
        s = ET.SubElement(root, "shipment")
        ET.SubElement(s, "shipment_id").text = str(i)
        ET.SubElement(s, "warehouse").text = ["W1", "W2"][i % 2]
        ET.SubElement(s, "created_at").text = f"2022-02-{(i % 28) + 1:02d}T00:00:00"
    ET.ElementTree(root).write(path)


# --------------------------------------------------------------------------
# iter_xml_batches
# --------------------------------------------------------------------------


def test_multi_batch_streaming(tmp_path):
    path = tmp_path / "shipments.xml"
    _write_shipments(path, 25)
    batches = list(iter_xml_batches(path, batch_rows=10))
    assert [b.height for b in batches] == [10, 10, 5]
    merged = pl.concat(batches, how="diagonal")
    assert merged.height == 25
    assert merged["shipment_id"].to_list() == [str(i) for i in range(25)]


def test_ragged_columns_across_batches(tmp_path):
    # `notes` only present on records past the first batch boundary --
    # optional leaves may be entirely absent from an early batch.
    root = ET.Element("shipments")
    for i in range(6):
        s = ET.SubElement(root, "shipment")
        ET.SubElement(s, "shipment_id").text = str(i)
        if i >= 3:
            ET.SubElement(s, "notes").text = "late"
    path = tmp_path / "shipments.xml"
    ET.ElementTree(root).write(path)

    batches = list(iter_xml_batches(path, batch_rows=3))
    assert "notes" not in batches[0].columns
    assert "notes" in batches[1].columns

    merged = pl.concat(batches, how="diagonal")
    assert merged.height == 6
    assert merged["notes"].null_count() == 3


def test_all_string_dtypes(tmp_path):
    path = tmp_path / "shipments.xml"
    _write_shipments(path, 5)
    for batch in iter_xml_batches(path, batch_rows=2):
        assert all(dt == pl.String for dt in batch.dtypes)


def test_first_seen_column_order(tmp_path):
    root = ET.Element("recs")
    for i in range(3):
        r = ET.SubElement(root, "rec")
        ET.SubElement(r, "zeta").text = str(i)
        ET.SubElement(r, "alpha").text = str(i)
        ET.SubElement(r, "mid").text = str(i)
    path = tmp_path / "recs.xml"
    ET.ElementTree(root).write(path)

    (batch,) = list(iter_xml_batches(path, batch_rows=100))
    assert batch.columns == ["zeta", "alpha", "mid"]


def test_nested_container_flattening_and_collision(tmp_path):
    root = ET.Element("things")
    for i in range(4):
        t = ET.SubElement(root, "thing")
        ET.SubElement(t, "id").text = str(i)
        nested = ET.SubElement(t, "nested")
        ET.SubElement(nested, "id").text = str(i * 2)
        ET.SubElement(nested, "label").text = "x"
    path = tmp_path / "things.xml"
    ET.ElementTree(root).write(path)

    (batch,) = list(iter_xml_batches(path, batch_rows=100))
    # "id" collides between the top-level leaf and the nested struct's leaf
    # -> the nested one falls back to {owner_tag}_{leaf} = "nested_id".
    assert set(batch.columns) == {"id", "nested_id", "label"}
    assert batch["id"].to_list() == ["0", "1", "2", "3"]
    assert batch["nested_id"].to_list() == ["0", "2", "4", "6"]


def test_namespace_local_names(tmp_path):
    xml_text = (
        '<?xml version="1.0"?>'
        '<ns:shipments xmlns:ns="urn:example">'
        "<ns:shipment><ns:shipment_id>1</ns:shipment_id>"
        "<ns:warehouse>W1</ns:warehouse></ns:shipment>"
        "</ns:shipments>"
    )
    path = tmp_path / "ns.xml"
    path.write_text(xml_text)

    (batch,) = list(iter_xml_batches(path, batch_rows=100))
    assert set(batch.columns) == {"shipment_id", "warehouse"}
    assert batch["shipment_id"].to_list() == ["1"]


def test_empty_root_zero_batches(tmp_path):
    path = tmp_path / "empty.xml"
    path.write_text("<shipments></shipments>")
    assert list(iter_xml_batches(path)) == []


def test_reference_backend_iter_batches_matches_dispatch(tmp_path, monkeypatch):
    """Exercise the reference-specific internals directly (memory-behavior
    parity check) and confirm the public dispatch agrees."""
    path = tmp_path / "shipments.xml"
    _write_shipments(path, 12)
    monkeypatch.setattr(xmlstream, "_RUST", None)

    ref_batches = list(xmlstream._iter_batches_reference(path, batch_rows=5))
    assert [b.height for b in ref_batches] == [5, 5, 2]

    dispatched = list(iter_xml_batches(path, batch_rows=5))
    assert [b.height for b in dispatched] == [5, 5, 2]


# --------------------------------------------------------------------------
# count_xml_records
# --------------------------------------------------------------------------


def test_count_matches_streamed_total(tmp_path):
    path = tmp_path / "shipments.xml"
    _write_shipments(path, 37)
    assert count_xml_records(path) == 37
    assert sum(b.height for b in iter_xml_batches(path, batch_rows=7)) == 37


# --------------------------------------------------------------------------
# xml_to_parquet
# --------------------------------------------------------------------------


def test_xml_to_parquet_parts_and_total(tmp_path):
    path = tmp_path / "shipments.xml"
    _write_shipments(path, 25)
    out_dir = tmp_path / "out"
    total = xml_to_parquet(path, out_dir, batch_rows=10)
    assert total == 25

    parts = sorted(p.name for p in out_dir.glob("*.parquet"))
    assert parts == [
        "part-0000000-0000.parquet",
        "part-0000000-0001.parquet",
        "part-0000000-0002.parquet",
    ]

    con = duckdb.connect()
    try:
        (count,) = con.execute(
            f"SELECT count(*) FROM read_parquet('{out_dir}/*.parquet', union_by_name=true)"
        ).fetchone()
    finally:
        con.close()
    assert count == 25


# --------------------------------------------------------------------------
# xml_dir_to_parquet
# --------------------------------------------------------------------------


def _write_records_file(path: Path, ids: range, extra: bool) -> None:
    root = ET.Element("recs")
    for i in ids:
        r = ET.SubElement(root, "rec")
        ET.SubElement(r, "id").text = str(i)
        # Always a decimal string -> must infer DOUBLE, never BIGINT (the
        # BIGINT candidate is guarded by a pure-integer regex in xmlstream).
        ET.SubElement(r, "ratio").text = "0.5"
        ET.SubElement(r, "active").text = "true" if i % 2 == 0 else "false"
        ET.SubElement(r, "created_at").text = f"2022-01-{(i % 28) + 1:02d}T00:00:00"
        ET.SubElement(r, "label").text = f"rec-{i}"
        if extra:
            ET.SubElement(r, "extra").text = str(i * 10)
    ET.ElementTree(root).write(path)


def _make_ragged_dir(tmp_path: Path) -> tuple[Path, int]:
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    n_files = 30
    per_file = 3
    total = 0
    for fi in range(n_files):
        ids = range(fi * per_file, fi * per_file + per_file)
        extra = fi >= 15  # "extra" column only appears in later files
        _write_records_file(in_dir / f"f{fi:03d}.xml", ids, extra)
        total += per_file
    return in_dir, total


def test_xml_dir_to_parquet_typed(tmp_path):
    in_dir, total = _make_ragged_dir(tmp_path)
    out_dir = tmp_path / "out"
    n = xml_dir_to_parquet(in_dir, out_dir, workers=2, batch_rows=1000, infer_types=True)
    assert n == total

    con = duckdb.connect()
    try:
        rows = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{out_dir}/*.parquet', union_by_name=true)"
        ).fetchall()
        types = {r[0]: r[1] for r in rows}
        (count,) = con.execute(
            f"SELECT count(*) FROM read_parquet('{out_dir}/*.parquet', union_by_name=true)"
        ).fetchone()
    finally:
        con.close()

    assert count == total
    assert types["id"] == "BIGINT"
    assert types["ratio"] == "DOUBLE"  # decimal string, not BIGINT
    assert types["active"] == "BOOLEAN"
    assert types["created_at"] == "TIMESTAMP"
    assert types["label"] == "VARCHAR"
    assert types["extra"] == "BIGINT"


def test_xml_dir_to_parquet_untyped_all_varchar(tmp_path):
    in_dir, total = _make_ragged_dir(tmp_path)
    out_dir = tmp_path / "out"
    n = xml_dir_to_parquet(in_dir, out_dir, workers=2, batch_rows=1000, infer_types=False)
    assert n == total

    con = duckdb.connect()
    try:
        rows = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{out_dir}/*.parquet', union_by_name=true)"
        ).fetchall()
    finally:
        con.close()
    assert all(r[1] == "VARCHAR" for r in rows)


def test_xml_dir_to_parquet_empty_dir_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        xml_dir_to_parquet(empty, tmp_path / "out")


def test_xml_dir_to_parquet_single_file_default_workers(tmp_path):
    in_dir = tmp_path / "single"
    in_dir.mkdir()
    _write_records_file(in_dir / "only.xml", range(5), extra=False)
    out_dir = tmp_path / "out"
    n = xml_dir_to_parquet(in_dir, out_dir)
    assert n == 5


# --------------------------------------------------------------------------
# CLI: `verisynth ingest`
# --------------------------------------------------------------------------


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "verisynth.cli", *args],
        capture_output=True,
        text=True,
    )


def test_cli_ingest_dir_and_file(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for fi in range(4):
        _write_shipments(in_dir / f"f{fi}.xml", 5, start=fi * 5)

    out_dir = tmp_path / "staged"
    result = _run_cli(
        "ingest",
        "--input", str(in_dir),
        "--out", str(out_dir),
        "--table", "shipments",
        "--workers", "2",
    )
    assert result.returncode == 0, result.stderr
    assert "20 records" in result.stdout

    con = duckdb.connect()
    try:
        (count,) = con.execute(
            f"SELECT count(*) FROM read_parquet("
            f"'{out_dir}/shipments/*.parquet', union_by_name=true)"
        ).fetchone()
    finally:
        con.close()
    assert count == 20

    single_path = tmp_path / "one.xml"
    _write_shipments(single_path, 7)
    result = _run_cli(
        "ingest",
        "--input", str(single_path),
        "--out", str(out_dir),
        "--table", "single",
    )
    assert result.returncode == 0, result.stderr
    assert "7 records" in result.stdout

    con = duckdb.connect()
    try:
        (count2,) = con.execute(
            f"SELECT count(*) FROM read_parquet('{out_dir}/single/*.parquet')"
        ).fetchone()
    finally:
        con.close()
    assert count2 == 7


def test_cli_ingest_missing_input_errors(tmp_path):
    result = _run_cli(
        "ingest",
        "--input", str(tmp_path / "nope"),
        "--out", str(tmp_path / "out"),
        "--table", "x",
    )
    assert result.returncode != 0
    assert "not found" in result.stderr
