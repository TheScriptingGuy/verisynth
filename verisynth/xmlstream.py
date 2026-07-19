"""Streaming XML: bounded-memory record reading and batch ingestion to Parquet.

See docs/ARCHITECTURE.md §12. XML files up to multi-GB are read as a stream
of flattened record batches — memory stays O(batch), never O(file) — and
directories of many XML files (100k+) are ingested in parallel into a
Parquet staging dataset that DuckDB/scan/fit consume out-of-core.

Backend dispatch mirrors kernels.py: the Rust fast path
(``verisynth_kernels.stream_xml_file`` / ``count_xml_records``, quick-xml)
is used when the extension is built with XML support; otherwise a pure
Python ``xml.etree.ElementTree.iterparse`` fallback with **identical record
semantics** runs (parity-tested). Set ``VERISYNTH_FORCE_REFERENCE=1`` to
force the fallback.

Record semantics (normative — the Rust port must match exactly):

- A record is each direct child element of the document root.
- Within a record, children are walked in document order. An element with
  child elements recurses (like a JSON struct); an element without children
  is a leaf contributing its whitespace-stripped text (empty -> null).
- Flattened keys prefer the leaf tag name; on collision with an
  already-populated key the value is stored under ``{owner_tag}_{leaf}``
  where ``owner_tag`` is the tag of the element that directly owns the
  colliding field (the same rule as JSON struct flattening in scanner.py).
- Tag names are reduced to their **local name**: ElementTree's expanded
  ``{uri}tag`` and quick-xml's raw ``prefix:tag`` both become ``tag``, so
  the two backends agree and namespaced documents yield clean column names.
- Element attributes and mixed text content of container elements are
  ignored.
"""

from __future__ import annotations

import multiprocessing
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator

import duckdb
import polars as pl

DEFAULT_BATCH_ROWS = 65536

BACKEND: str


# --------------------------------------------------------------------------
# Record flattening (canonical implementation; scanner.py imports this)
# --------------------------------------------------------------------------


def _local_tag(tag: str) -> str:
    """Local name: strips ET's expanded ``{uri}`` prefix (and any ``ns:``
    prefix, for parity with the raw-name Rust backend)."""
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def flatten_xml_record(elem: ET.Element) -> dict[str, str | None]:
    """Flatten one XML record element into a leaf-named dict of raw strings."""
    out: dict[str, str | None] = {}
    for child in elem:
        if not isinstance(child.tag, str):
            continue  # comments / processing instructions
        if list(child):
            sub = flatten_xml_record(child)
            for k, v in sub.items():
                key = k if k not in out else f"{_local_tag(child.tag)}_{k}"
                out[key] = v
        else:
            text = (child.text or "").strip()
            tag = _local_tag(child.tag)
            key = tag if tag not in out else f"{_local_tag(elem.tag)}_{tag}"
            out[key] = text if text else None
    return out


# --------------------------------------------------------------------------
# Reference backend: ET.iterparse streaming
# --------------------------------------------------------------------------


def _iter_records_reference(path: str | Path) -> Iterator[dict[str, str | None]]:
    """Stream flattened records from ``path`` with O(record) memory.

    Finished records are cleared and detached from the root element so the
    tree never accumulates.
    """
    root: ET.Element | None = None
    depth = 0
    for event, elem in ET.iterparse(str(path), events=("start", "end")):
        if event == "start":
            if root is None:
                root = elem
            depth += 1
        else:
            depth -= 1
            if depth == 1 and elem is not root:
                yield flatten_xml_record(elem)
                elem.clear()
                # Detach the finished record (and any predecessors) from the
                # root so memory stays bounded on multi-GB files.
                if root is not None and len(root):
                    del root[:]


def iter_xml_record_elements(path: str | Path) -> Iterator[ET.Element]:
    """Stream the RAW record elements (direct children of the root) with
    O(record) memory — for consumers that need the unflattened subtree, e.g.
    the scanner's nested-entity extraction. Always uses the ElementTree
    backend (callers cap consumption; the flattened fast path stays in
    ``iter_xml_batches``). The yielded element is detached and cleared after
    the consumer's iteration step, so it must not be retained."""
    root: ET.Element | None = None
    depth = 0
    for event, elem in ET.iterparse(str(path), events=("start", "end")):
        if event == "start":
            if root is None:
                root = elem
            depth += 1
        else:
            depth -= 1
            if depth == 1 and elem is not root:
                yield elem
                elem.clear()
                if root is not None and len(root):
                    del root[:]


def _count_records_reference(path: str | Path) -> int:
    root: ET.Element | None = None
    depth = 0
    n = 0
    for event, elem in ET.iterparse(str(path), events=("start", "end")):
        if event == "start":
            if root is None:
                root = elem
            depth += 1
        else:
            depth -= 1
            if depth == 1 and elem is not root:
                n += 1
                elem.clear()
                if root is not None and len(root):
                    del root[:]
    return n


def _records_to_frame(records: list[dict[str, str | None]]) -> pl.DataFrame:
    """Batch of flattened records -> all-string DataFrame (column order =
    first-seen order across the batch, matching dict-union semantics)."""
    df = pl.DataFrame(records, infer_schema_length=None)
    # All-null columns infer as Null dtype; force String so batches concat
    # and staged Parquet parts union cleanly.
    return df.with_columns(pl.all().cast(pl.String))


def _iter_batches_reference(
    path: str | Path, batch_rows: int
) -> Iterator[pl.DataFrame]:
    records: list[dict[str, str | None]] = []
    for rec in _iter_records_reference(path):
        records.append(rec)
        if len(records) >= batch_rows:
            yield _records_to_frame(records)
            records = []
    if records:
        yield _records_to_frame(records)


# --------------------------------------------------------------------------
# Backend dispatch (kernels.py pattern)
# --------------------------------------------------------------------------


def _rust_backend():
    """Return the Rust module if it is importable AND carries the XML
    streaming functions (an older kernels wheel may predate them)."""
    if os.environ.get("VERISYNTH_FORCE_REFERENCE") == "1":
        return None
    try:
        import verisynth_kernels as _rust
    except ImportError:
        return None
    if hasattr(_rust, "stream_xml_file") and hasattr(_rust, "count_xml_records"):
        return _rust
    return None


_RUST = _rust_backend()
BACKEND = "rust" if _RUST is not None else "reference"


def iter_xml_batches(
    path: str | Path, batch_rows: int = DEFAULT_BATCH_ROWS
) -> Iterator[pl.DataFrame]:
    """Stream ``path`` as flattened, all-string DataFrames of ``<= batch_rows``
    rows each. Memory is O(batch) regardless of file size."""
    if batch_rows <= 0:
        raise ValueError(f"batch_rows must be > 0 (got {batch_rows})")
    if _RUST is not None:
        for names, columns in _RUST.stream_xml_file(str(path), batch_rows):
            data = dict(zip(names, columns))
            yield pl.DataFrame(data, schema={n: pl.String for n in names})
    else:
        yield from _iter_batches_reference(path, batch_rows)


def count_xml_records(path: str | Path) -> int:
    """Number of records (direct children of the root) in ``path``,
    streamed — never builds the document tree."""
    if _RUST is not None:
        return int(_RUST.count_xml_records(str(path)))
    return _count_records_reference(path)


# --------------------------------------------------------------------------
# Parquet staging: single file and 100k-file batch ingestion
# --------------------------------------------------------------------------


def xml_to_parquet(
    path: str | Path,
    out_dir: str | Path,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    part_prefix: str = "part-0000000",
) -> int:
    """Stream one XML file into ``{out_dir}/{part_prefix}-{batch:04d}.parquet``
    string-typed parts. Returns the number of records written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, df in enumerate(iter_xml_batches(path, batch_rows)):
        df.write_parquet(out / f"{part_prefix}-{i:04d}.parquet")
        total += df.height
    return total


def _ingest_one(args: tuple[str, str, int, int]) -> int:
    path, out_dir, file_idx, batch_rows = args
    return xml_to_parquet(
        path, out_dir, batch_rows=batch_rows, part_prefix=f"part-{file_idx:07d}"
    )


# Cast targets mirroring scanner._infer_xml_column's order: int64, then
# float64, then bool (true/false), then timestamp; else the column stays
# text. Each entry is (type, cast expression, per-value failure predicate).
# BIGINT needs the pure-integer regex guard: DuckDB's TRY_CAST rounds
# decimal strings ('0.5' -> 1) where the polars strict cast this mirrors
# would fail.
_INT_RE = "^[+-]?[0-9]+$"
_CAST_CANDIDATES: tuple[tuple[str, str, str], ...] = (
    (
        "BIGINT",
        "TRY_CAST({c} AS BIGINT)",
        "(NOT regexp_matches({c}, '" + _INT_RE + "') OR TRY_CAST({c} AS BIGINT) IS NULL)",
    ),
    ("DOUBLE", "TRY_CAST({c} AS DOUBLE)", "TRY_CAST({c} AS DOUBLE) IS NULL"),
    (
        "BOOLEAN",
        "CASE WHEN {c} = 'true' THEN true WHEN {c} = 'false' THEN false END",
        "{c} NOT IN ('true', 'false')",
    ),
    ("TIMESTAMP", "TRY_CAST({c} AS TIMESTAMP)", "TRY_CAST({c} AS TIMESTAMP) IS NULL"),
)


def _finalize_types(strings_glob: str, out_dir: Path) -> None:
    """One out-of-core DuckDB pass over the string staging: per column, pick
    the first cast target that succeeds for every non-null value (same
    preference order as scanner's XML type inference), then rewrite the
    dataset with those casts applied."""
    con = duckdb.connect()
    try:
        cols = [
            r[0]
            for r in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{strings_glob}', union_by_name=true)"
            ).fetchall()
        ]
        exprs: list[str] = []
        for c in cols:
            qc = f'"{c}"'
            chosen = qc
            for _sql_type, cast_tpl, fail_tpl in _CAST_CANDIDATES:
                probe = f"count(*) FILTER ({qc} IS NOT NULL AND {fail_tpl.format(c=qc)})"
                (bad,) = con.execute(
                    f"SELECT {probe} FROM read_parquet('{strings_glob}', union_by_name=true)"
                ).fetchone()
                if bad == 0:
                    chosen = f"{cast_tpl.format(c=qc)} AS {qc}"
                    break
            exprs.append(chosen)
        con.execute(
            f"COPY (SELECT {', '.join(exprs)} FROM "
            f"read_parquet('{strings_glob}', union_by_name=true)) "
            f"TO '{out_dir}' (FORMAT PARQUET, PER_THREAD_OUTPUT)"
        )
    finally:
        con.close()


def xml_dir_to_parquet(
    input_dir: str | Path,
    out_dir: str | Path,
    workers: int | None = None,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    infer_types: bool = True,
) -> int:
    """Ingest every ``*.xml`` file under ``input_dir`` (sorted, one logical
    table) into a Parquet dataset at ``out_dir``, in parallel.

    Each file is streamed independently by a worker process (memory per
    worker is O(batch)); schemas may be ragged across files — consumers read
    with ``union_by_name``. With ``infer_types`` (default) a final
    out-of-core DuckDB pass applies scanner-compatible column typing
    (int64 -> float64 -> bool -> timestamp -> string) and compacts the
    dataset; with ``infer_types=False`` the raw all-string parts land in
    ``out_dir`` directly. Returns the total record count.
    """
    in_dir = Path(input_dir)
    files = sorted(p for p in in_dir.glob("*.xml") if p.is_file())
    if not files:
        raise FileNotFoundError(f"xml_dir_to_parquet: no .xml files found in {in_dir}")

    out = Path(out_dir)
    # Type inference stages raw strings in a sibling dir: DuckDB's COPY TO
    # directory requires an empty target for the final typed rewrite.
    staging = out.parent / f"{out.name}._strings" if infer_types else out
    staging.mkdir(parents=True, exist_ok=True)

    tasks = [
        (str(p), str(staging), i, batch_rows) for i, p in enumerate(files)
    ]
    if len(files) == 1 or workers == 1:
        total = sum(_ingest_one(t) for t in tasks)
    else:
        max_workers = workers or os.cpu_count() or 4
        chunksize = max(1, len(tasks) // (max_workers * 8))
        # spawn, not fork: polars/duckdb threads in the parent make forked
        # children prone to deadlock.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
            total = sum(pool.map(_ingest_one, tasks, chunksize=chunksize))

    if infer_types:
        if total > 0:
            _finalize_types(str(staging / "*.parquet"), out)
        import shutil

        shutil.rmtree(staging)
    return total
