"""Metadata scanner: profile real tables and detect structure.

Reads a directory of ``{table}.parquet`` / ``{table}.csv`` / ``{table}.json``
(or ``.jsonl`` / ``.ndjson``) / ``{table}.xml`` files -- plus ``{table}/``
subdirectories containing one or more ``*.xml`` files, loaded as a single
logical table -- and infers the structural facts a metadata skeleton needs:
column types and null rates, primary-key candidates, foreign-key relations
(with value-containment evidence), parent/child cardinality profiles, and
per-column distribution suggestions. The interactive wizard (``verisynth
init``, see wizard.py) uses the report as chat suggestions; ``verisynth
scan`` prints it directly.

Detection is heuristic and advisory: nothing here mutates data, and every
finding carries the evidence (coverage, counts) so a human can veto it.
Every loaded frame is also post-processed by ``_expand_document_columns``,
which sniffs and flattens string columns holding a JSON object or XML
fragment per row (the scan-side counterpart of the metadata ``document:``
column spec) before profiling.
"""

from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import duckdb
import polars as pl

from ._skeleton_infer import assign_source, infer_skeleton, init_from_dir  # noqa: F401
from .xmlstream import flatten_xml_record, iter_xml_record_elements

# Scan is advisory profiling, not a full ingestion pipeline (see `verisynth
# ingest` / xmlstream.xml_dir_to_parquet for that): cap the number of XML
# records materialized for a table at this many rows so a multi-GB file or
# huge xml/ subdirectory can still be profiled in bounded time/memory,
# overridable via VERISYNTH_XML_SCAN_ROWS for tests/tuning.
_XML_SCAN_MAX_ROWS = 1_000_000

# Containment threshold for accepting a foreign-key candidate: at least this
# fraction of the child column's non-null values must exist in the parent key.
# Shared with the deterministic inference layer (_skeleton_infer.py, TASK
# CARD 16 rule 2).
FK_COVERAGE_THRESHOLD = 0.98

# Max distinct values for a categorical distribution suggestion.
_CATEGORICAL_MAX = 20


@dataclass
class ColumnProfile:
    name: str
    type: str  # metadata DSL type: int64 | float64 | string | bool | timestamp
    null_rate: float
    n_unique: int  # distinct non-null values
    unique: bool  # no nulls and every value distinct
    suggestion: dict[str, Any] | None = None  # suggested distribution spec


@dataclass
class Relation:
    child: str
    child_column: str
    parent: str
    parent_key: str
    coverage: float  # fraction of child non-null values present in parent key
    cardinality: dict[str, Any]  # suggested cardinality spec (kind + params)
    child_stride: int
    mean_children: float
    max_children: int


@dataclass
class TableScan:
    name: str
    rows: int
    columns: dict[str, ColumnProfile]
    pk: str | None
    pk_candidates: list[str] = field(default_factory=list)


@dataclass
class ScanReport:
    tables: dict[str, TableScan]
    relations: list[Relation]

    def relations_of(self, child: str) -> list[Relation]:
        return [r for r in self.relations if r.child == child]


# --------------------------------------------------------------------------
# Loading & column profiling
# --------------------------------------------------------------------------


def _load_frames(input_dir: str | Path) -> dict[str, pl.DataFrame]:
    p = Path(input_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"scan: {p} is not a directory")
    frames: dict[str, pl.DataFrame] = {}
    def _add_xml(name: str, loaded: tuple[pl.DataFrame, dict[str, pl.DataFrame]]) -> None:
        df, children = loaded
        frames[name] = df
        # Nested XML entity collections become their own tables, named after
        # the repeated/container tag ({parent}_{tag} on collision).
        for tag, cdf in children.items():
            frames[tag if tag not in frames else f"{name}_{tag}"] = cdf

    for f in sorted(p.iterdir()):
        if f.is_dir():
            # A subdirectory of *.xml files is one logical table named after
            # the subdirectory (100k-file ingestion uses the same shape --
            # see xmlstream.xml_dir_to_parquet -- this is the scan-side
            # analogue, capped for profiling).
            xml_files = sorted(x for x in f.glob("*.xml") if x.is_file())
            if xml_files:
                _add_xml(f.name, _load_xml_dir(xml_files))
            continue
        if f.suffix == ".parquet":
            frames[f.stem] = pl.read_parquet(f)
        elif f.suffix == ".csv":
            frames[f.stem] = pl.read_csv(f, try_parse_dates=True)
        elif f.suffix in (".json", ".jsonl", ".ndjson"):
            frames[f.stem] = _flatten_structs(_load_json(f))
        elif f.suffix == ".xml":
            _add_xml(f.stem, _load_xml(f))
    if not frames:
        raise FileNotFoundError(
            f"scan: no .parquet, .csv, .json/.jsonl/.ndjson or .xml data files "
            f"(or {{table}}/ xml subdirectories) found in {p}"
        )
    # In-table document columns (a string column storing a JSON object / XML
    # fragment per row) apply to every input format alike -- expand them
    # after loading, regardless of how the frame reached us. Then extract
    # nested entity collections (JSON list-of-struct columns) into separate
    # child tables so PK/FK detection can recover the relation.
    frames = {name: _expand_document_columns(df) for name, df in frames.items()}
    return _extract_nested_tables(frames)


def _rescue_datetime_strings(df: pl.DataFrame) -> pl.DataFrame:
    """Rescue string columns that parse cleanly as ISO-8601 (T-separated)
    datetimes into ``pl.Datetime``, mirroring ``read_csv``'s
    ``try_parse_dates``. Shared by ``_load_json`` (DuckDB's sniffer leaves
    T-separated timestamps as VARCHAR) and ``_expand_document_columns``'s
    JSON payload expansion (§11)."""
    for c in df.columns:
        if df[c].dtype == pl.String:
            try:
                df = df.with_columns(df[c].str.to_datetime(time_unit="us"))
            except pl.exceptions.PolarsError:
                pass
    return df


def _load_json(path: Path) -> pl.DataFrame:
    """Load a JSON array or newline-delimited-JSON file via DuckDB.

    ``read_json_auto`` handles both shapes (and nested objects, which surface
    as ``pl.Struct`` columns that ``_flatten_structs`` unnests below) and
    auto-detects timestamp-shaped strings.
    """
    con = duckdb.connect()
    try:
        arrow_tbl = con.execute(f"SELECT * FROM read_json_auto('{path}')").arrow()
    finally:
        con.close()
    df = pl.from_arrow(arrow_tbl)
    # DuckDB's sniffer types space-separated timestamps but leaves
    # T-separated ISO-8601 strings (the JSON-document convention, §11) as
    # VARCHAR -- rescue string columns that parse cleanly as datetimes,
    # mirroring read_csv's try_parse_dates.
    return _rescue_datetime_strings(df)


def _flatten_structs(df: pl.DataFrame) -> pl.DataFrame:
    """Recursively unnest ``pl.Struct`` columns into scalar leaf columns.

    Each struct field is promoted to a top-level column under its own leaf
    name; on a name collision with an already-existing column, the field is
    renamed to ``{struct_column}_{field}`` instead. Repeats until no struct
    columns remain (handles structs nested inside structs).
    """
    while True:
        struct_col = next((c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Struct), None)
        if struct_col is None:
            return df
        existing = set(df.columns) - {struct_col}
        field_names = [fld.name for fld in df.schema[struct_col].fields]
        seen: set[str] = set()
        new_names = []
        for fname in field_names:
            new_name = fname if fname not in existing and fname not in seen else f"{struct_col}_{fname}"
            seen.add(new_name)
            new_names.append(new_name)
        df = df.with_columns(pl.col(struct_col).struct.rename_fields(new_names)).unnest(struct_col)


# --------------------------------------------------------------------------
# XML loading
# --------------------------------------------------------------------------

# Canonical flattening implementation now lives in xmlstream.py (streaming
# reader + Rust/reference backend parity); kept here as a backward-compat
# alias for any external callers of the old private name.
_flatten_xml_record = flatten_xml_record


def _infer_xml_column(s: pl.Series) -> pl.Series:
    """Type an all-string XML column: int64, else float64, else bool, else
    timestamp, else leave as string (in that order of preference)."""
    nn = s.drop_nulls()
    if nn.len() == 0:
        return s
    try:
        return s.cast(pl.Int64, strict=True)
    except pl.exceptions.PolarsError:
        pass
    try:
        return s.cast(pl.Float64, strict=True)
    except pl.exceptions.PolarsError:
        pass
    if set(nn.unique().to_list()) <= {"true", "false"}:
        return s.replace_strict({"true": True, "false": False}, default=None, return_dtype=pl.Boolean)
    try:
        return s.str.to_datetime(time_unit="us")
    except pl.exceptions.PolarsError:
        pass
    return s


# --------------------------------------------------------------------------
# In-table document columns (§11): a string column storing a JSON object /
# XML fragment per row, sniffed and flattened during scan-side profiling.
# The generation-side counterpart is the metadata ``document:`` column spec
# (``ColumnDocumentSpec`` in metadata.py) -- fit.py's ``_expand_document_columns``
# is the skeleton-aware fitting variant of the same idea.
# --------------------------------------------------------------------------

#: How many non-null values of a string column to sniff before deciding
#: whether it holds a JSON object or XML fragment per row.
_DOCUMENT_SNIFF_SAMPLE = 100


def _looks_like_json_payload(sample: list[str]) -> bool:
    if not sample:
        return False
    for v in sample:
        s = v.strip()
        if not s.startswith("{"):
            return False
        try:
            obj = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return False
        if not isinstance(obj, dict):
            return False
    return True


def _looks_like_xml_payload(sample: list[str]) -> bool:
    if not sample:
        return False
    for v in sample:
        s = v.strip()
        if not s.startswith("<"):
            return False
        try:
            ET.fromstring(s)
        except ET.ParseError:
            return False
    return True


def _merge_document_expansion(df: pl.DataFrame, payload_col: str, sub: pl.DataFrame) -> pl.DataFrame:
    """Merge ``sub`` (one column per flattened leaf, same row order/height as
    ``df``) into ``df``, dropping ``payload_col``. Leaf names collide with an
    existing column (or another leaf of the same expansion) fall back to
    ``{payload_col}_{leaf}`` -- the same convention as ``_flatten_structs``.
    """
    existing = set(df.columns) - {payload_col}
    seen: set[str] = set()
    rename: dict[str, str] = {}
    for leaf in sub.columns:
        new_name = leaf if leaf not in existing and leaf not in seen else f"{payload_col}_{leaf}"
        seen.add(new_name)
        rename[leaf] = new_name
    sub = sub.rename(rename)
    return df.drop(payload_col).hstack(sub)


def _expand_json_document_column(df: pl.DataFrame, payload_col: str) -> pl.DataFrame:
    dicts = [json.loads(v) if v is not None else None for v in df[payload_col].to_list()]
    sub = pl.DataFrame(dicts, infer_schema_length=None)
    sub = _flatten_structs(sub)
    sub = _rescue_datetime_strings(sub)
    return _merge_document_expansion(df, payload_col, sub)


def _expand_xml_document_column(df: pl.DataFrame, payload_col: str) -> pl.DataFrame:
    records = [
        flatten_xml_record(ET.fromstring(v)) if v is not None else None
        for v in df[payload_col].to_list()
    ]
    sub = pl.DataFrame(records, infer_schema_length=None)
    sub = sub.with_columns(pl.all().cast(pl.String))
    sub = sub.with_columns(_infer_xml_column(sub[c]) for c in sub.columns)
    return _merge_document_expansion(df, payload_col, sub)


def _expand_document_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Sniff every string column: if (up to) its first 100 non-null values
    all parse as a JSON object or an XML fragment, expand it into flattened,
    typed sibling columns and drop the payload column -- a unique-ish giant
    string column would otherwise pollute PK-candidate ranking. Columns with
    mixed/plain text (or no non-null values) are left untouched.
    """
    for cname in list(df.columns):
        s = df[cname]
        if s.dtype != pl.String:
            continue
        sample = s.drop_nulls().head(_DOCUMENT_SNIFF_SAMPLE).to_list()
        if _looks_like_json_payload(sample):
            df = _expand_json_document_column(df, cname)
        elif _looks_like_xml_payload(sample):
            df = _expand_xml_document_column(df, cname)
    return df


# --------------------------------------------------------------------------
# Nested entity extraction (docs/ARCHITECTURE.md §11, read side of
# format.nest): a JSON list-of-struct column, or repeated XML child
# elements, hold the rows of a RELATED child entity -- profile them as a
# separate table with the parent's id-like columns injected so FK
# containment detection recovers the relation.
# --------------------------------------------------------------------------


def _id_like_columns(df: pl.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if (c == "id" or c.endswith("_id"))
        and not isinstance(df.schema[c], (pl.List, pl.Struct))
    ]


def _extract_nested_tables(frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Pull every list-of-struct column out of every frame into its own
    child-table frame (named after the column; ``{parent}_{column}`` on
    collision), exploded one row per nested record, with the parent's
    id-like columns injected (unless the nested records already carry
    them). Recursive: extracted children are re-examined for deeper
    nesting. The list column is dropped from the parent frame."""
    out = dict(frames)
    queue = list(frames.items())
    while queue:
        name, df = queue.pop(0)
        for cname in list(df.columns):
            dt = df.schema[cname]
            if not (isinstance(dt, pl.List) and isinstance(dt.inner, pl.Struct)):
                continue
            ids = [c for c in _id_like_columns(df) if c != cname]
            child = (
                df.select(ids + [cname])
                .explode(cname)
                .filter(pl.col(cname).is_not_null())
            )
            # _flatten_structs unnests the struct column; nested-record
            # fields colliding with the injected id columns land under
            # {column}_{field} per the shared convention.
            child = _rescue_datetime_strings(_flatten_structs(child))
            child_name = cname if cname not in out else f"{name}_{cname}"
            out[child_name] = child
            queue.append((child_name, child))
            df = df.drop(cname)
            out[name] = df
    return out


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _is_xml_entity_container(e: ET.Element) -> bool:
    """A container of nested entity records: element children only, every
    child itself has element children, and all children share one tag
    (e.g. ``<lines><line>..</line><line>..</line></lines>``)."""
    inner = [c for c in e if isinstance(c.tag, str)]
    return (
        bool(inner)
        and all(list(c) for c in inner)
        and len({_local_tag(c.tag) for c in inner}) == 1
    )


def _split_xml_record(
    elem: ET.Element,
) -> tuple[ET.Element, dict[str, list[dict[str, str | None]]]]:
    """Split one raw record element into (scalar shell, nested entities).

    Nested entities are (a) a tag repeated more than once where every
    occurrence has element children, or (b) a single container element
    matching ``_is_xml_entity_container`` (its grandchildren are the
    records, named after the container). Everything else stays in the
    scalar shell and flattens as before.
    """
    groups: dict[str, list[ET.Element]] = {}
    for child in elem:
        if isinstance(child.tag, str):
            groups.setdefault(_local_tag(child.tag), []).append(child)

    shell = ET.Element(elem.tag)
    nested: dict[str, list[dict[str, str | None]]] = {}
    for tag, els in groups.items():
        if len(els) > 1 and all(list(e) for e in els):
            nested[tag] = [flatten_xml_record(e) for e in els]
        elif len(els) == 1 and _is_xml_entity_container(els[0]):
            nested[tag] = [
                flatten_xml_record(c) for c in els[0] if isinstance(c.tag, str)
            ]
        else:
            shell.extend(els)
    return shell, nested


def _typed_xml_frame(rows: list[dict[str, str | None]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows, infer_schema_length=None)
    df = df.with_columns(pl.all().cast(pl.String))
    return df.with_columns(_infer_xml_column(df[c]) for c in df.columns)


def _load_xml_records(paths: list[Path]) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    """Stream raw XML records from ``paths`` (sorted order, capped at the
    scan sample size), splitting out nested entity collections into child
    frames keyed by their tag."""
    cap = _xml_scan_max_rows()
    parent_rows: list[dict[str, str | None]] = []
    nested_rows: dict[str, list[dict[str, str | None]]] = {}
    for f in paths:
        for elem in iter_xml_record_elements(f):
            shell, nested = _split_xml_record(elem)
            scalars = flatten_xml_record(shell)
            parent_rows.append(scalars)
            ids = {k: v for k, v in scalars.items() if k == "id" or k.endswith("_id")}
            for tag, dicts in nested.items():
                bucket = nested_rows.setdefault(tag, [])
                for d in dicts:
                    injected = {k: v for k, v in ids.items() if k not in d}
                    bucket.append({**injected, **d})
            if len(parent_rows) >= cap:
                break
        if len(parent_rows) >= cap:
            break
    return (
        _typed_xml_frame(parent_rows),
        {tag: _typed_xml_frame(rows) for tag, rows in nested_rows.items()},
    )


def _xml_scan_max_rows() -> int:
    raw = os.environ.get("VERISYNTH_XML_SCAN_ROWS")
    return int(raw) if raw else _XML_SCAN_MAX_ROWS


def _load_xml(path: Path) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    """Load a single XML file: records are the child elements of the
    document root, streamed record-by-record (bounded memory), capped at
    the scan sample size, with nested entity collections split out as
    child frames."""
    return _load_xml_records([path])


def _load_xml_dir(files: list[Path]) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    """Load a ``{table}/`` subdirectory of XML files as one logical table:
    chain each file's records (sorted file order) through the same
    record/cap logic used for a single file."""
    return _load_xml_records(files)


def _dsl_type(dtype: pl.DataType) -> str:
    if dtype == pl.Boolean:
        return "bool"
    if dtype.is_integer():
        return "int64"
    if dtype.is_float():
        return "float64"
    if dtype.is_temporal():
        return "timestamp"
    return "string"


def _rounded_probs(counts: list[int]) -> list[float]:
    """Frequencies rounded to 6 decimals, adjusted so they sum to exactly 1."""
    total = sum(counts)
    probs = [round(c / total, 6) for c in counts[:-1]]
    probs.append(round(1.0 - sum(probs), 6))
    return probs


def _categorical_suggestion(s: pl.Series) -> dict[str, Any] | None:
    vc = s.drop_nulls().value_counts(sort=True).head(_CATEGORICAL_MAX)
    if vc.height == 0:
        return None
    categories = vc[s.name].to_list()
    probs = _rounded_probs(vc["count"].to_list())
    if any(p < 0 for p in probs):
        return None
    return {"kind": "categorical", "categories": categories, "probs": probs}


def _suggest_distribution(s: pl.Series, dsl_type: str, n_unique: int) -> dict[str, Any] | None:
    """Suggest a distribution spec for a plain (non-key) column."""
    nn = s.drop_nulls()
    if nn.len() == 0:
        return None

    if dsl_type in ("bool", "string"):
        return _categorical_suggestion(s)

    if dsl_type == "timestamp":
        start, end = nn.min(), nn.max()
        if start == end:
            return None
        return {
            "kind": "datetime_uniform",
            "start": start.isoformat(),
            "end": end.isoformat(),
        }

    if dsl_type == "int64":
        if n_unique <= _CATEGORICAL_MAX:
            return _categorical_suggestion(s)
        return {"kind": "uniform_int", "low": int(nn.min()), "high": int(nn.max())}

    if dsl_type == "float64":
        if bool((nn > 0).all()):
            logs = nn.log()
            sigma = float(logs.std() or 0.0)
            if sigma > 0:
                return {
                    "kind": "lognormal",
                    "mu": round(float(logs.mean()), 6),
                    "sigma": round(sigma, 6),
                }
        std = float(nn.std() or 0.0)
        if std <= 0:
            return None
        return {"kind": "normal", "mean": round(float(nn.mean()), 6), "std": round(std, 6)}

    return None


def _profile_table(name: str, df: pl.DataFrame) -> TableScan:
    columns: dict[str, ColumnProfile] = {}
    for cname in df.columns:
        s = df[cname]
        nulls = s.null_count()
        n_unique = s.drop_nulls().n_unique()
        dsl_type = _dsl_type(s.dtype)
        unique = nulls == 0 and n_unique == df.height and df.height > 0
        columns[cname] = ColumnProfile(
            name=cname,
            type=dsl_type,
            null_rate=round(nulls / df.height, 6) if df.height else 0.0,
            n_unique=n_unique,
            unique=unique,
            suggestion=_suggest_distribution(s, dsl_type, n_unique),
        )

    pk_candidates = _rank_pk_candidates(name, columns)
    return TableScan(
        name=name,
        rows=df.height,
        columns=columns,
        pk=pk_candidates[0] if pk_candidates else None,
        pk_candidates=pk_candidates,
    )


# --------------------------------------------------------------------------
# Primary-key detection
# --------------------------------------------------------------------------


def _singular(table: str) -> str:
    return table[:-1] if table.endswith("s") else table


def _pk_name_score(table: str, column: str) -> int:
    """Higher = more primary-key-like name for this table."""
    if column == "id":
        return 4
    if column in (f"{table}_id", f"{_singular(table)}_id"):
        return 3
    # inv_products -> product_id: match the singular of the last name segment.
    if column == f"{_singular(table.rsplit('_', 1)[-1])}_id":
        return 2
    if column.endswith("_id"):
        return 1
    return 0


def _rank_pk_candidates(table: str, columns: dict[str, ColumnProfile]) -> list[str]:
    unique_cols = [c for c in columns.values() if c.unique]
    positions = {name: i for i, name in enumerate(columns)}
    return [
        c.name
        for c in sorted(
            unique_cols,
            key=lambda c: (-_pk_name_score(table, c.name), positions[c.name]),
        )
    ]


# --------------------------------------------------------------------------
# Foreign keys, relations & cardinality
# --------------------------------------------------------------------------


def _containment(child_vals: pl.Series, parent_vals: pl.Series) -> float:
    """Fraction of non-null child values present in the parent key column."""
    nn = child_vals.drop_nulls()
    if nn.len() == 0:
        return 0.0
    if nn.dtype != parent_vals.dtype:
        try:
            nn = nn.cast(parent_vals.dtype)
        except pl.exceptions.PolarsError:
            return 0.0
    return float(nn.is_in(parent_vals.unique().implode()).mean())


def _next_pow2_above(n: int) -> int:
    """Smallest power of two strictly greater than n (the child_stride rule)."""
    return 1 << max(n, 1).bit_length()


def _cardinality_profile(
    child_df: pl.DataFrame, fk: str, parent_df: pl.DataFrame, parent_key: str
) -> tuple[dict[str, Any], int, float, int]:
    """Suggest a cardinality spec from observed children-per-parent counts.

    Counts include parents with zero children. Returns
    ``(spec, child_stride, mean, max)``.
    """
    per_parent = child_df.drop_nulls(fk).group_by(fk).len()
    observed = per_parent["len"].cast(pl.Int64)
    n_parents = parent_df.height
    n_zero = max(n_parents - per_parent.height, 0)

    total_children = int(observed.sum()) if observed.len() else 0
    mean = total_children / n_parents if n_parents else 0.0
    max_c = int(observed.max()) if observed.len() else 0
    min_c = 0 if n_zero > 0 else (int(observed.min()) if observed.len() else 0)

    if max_c <= 1:
        spec: dict[str, Any] = {"kind": "bernoulli", "p": round(mean, 6)}
        eff_max = 1
    elif n_zero == 0 and min_c == max_c:
        spec = {"kind": "fixed", "n": max_c}
        eff_max = max_c
    else:
        # Variance over all parents, zero-children parents included.
        sum_sq = float((observed**2).sum()) if observed.len() else 0.0
        var = sum_sq / n_parents - mean**2 if n_parents else 0.0
        if mean > 0 and abs(var / mean - 1.0) <= 0.5:
            spec = {"kind": "poisson", "lam": round(max(mean, 0.01), 6), "max": max_c}
        else:
            spec = {"kind": "uniform_int", "low": min_c, "high": max_c, "max": max_c}
        eff_max = max_c

    return spec, _next_pow2_above(eff_max), round(mean, 6), max_c


def _fk_candidate_parents(
    table: str, column: ColumnProfile, tables: dict[str, TableScan]
) -> list[str]:
    """Parent tables whose primary key this column could reference, by name.

    Shared rule with the deterministic inference layer (_skeleton_infer.py,
    TASK CARD 16 rule 2): a column is only a candidate if its name is
    literally the same as the other table's chosen primary key column.
    """
    out = []
    for pname, pscan in tables.items():
        if pname == table or pscan.pk is None:
            continue
        if column.name == pscan.pk:
            out.append(pname)
    return out


def _detect_relations(
    tables: dict[str, TableScan], frames: dict[str, pl.DataFrame]
) -> list[Relation]:
    relations: list[Relation] = []
    for tname, tscan in tables.items():
        for cname, col in tscan.columns.items():
            if cname == tscan.pk:
                continue
            for pname in _fk_candidate_parents(tname, col, tables):
                pk = tables[pname].pk
                coverage = _containment(frames[tname][cname], frames[pname][pk])
                if coverage < FK_COVERAGE_THRESHOLD:
                    continue
                spec, stride, mean, max_c = _cardinality_profile(
                    frames[tname], cname, frames[pname], pk
                )
                relations.append(
                    Relation(
                        child=tname,
                        child_column=cname,
                        parent=pname,
                        parent_key=pk,
                        coverage=round(coverage, 6),
                        cardinality=spec,
                        child_stride=stride,
                        mean_children=mean,
                        max_children=max_c,
                    )
                )
    return relations


def rank_parent_relations(table: str, relations: list[Relation]) -> list[Relation]:
    """Order a table's inbound relations by how parent-like each target is.

    A true parent tends to share a name stem with its child (``orders`` /
    ``order_items``) and to have few children per parent; a dimension
    reference (``product_id`` -> product catalog) has a high fan-out.
    """

    def key(r: Relation) -> tuple:
        stem_match = table.startswith(_singular(r.parent)) or r.parent.startswith(
            _singular(table).rsplit("_", 1)[0]
        )
        return (not stem_match, r.mean_children, -r.coverage)

    return sorted(relations, key=key)


# --------------------------------------------------------------------------
# Entry point & rendering
# --------------------------------------------------------------------------


def scan_directory(input_dir: str | Path) -> ScanReport:
    """Scan ``input_dir`` and return the full structural report."""
    frames = _load_frames(input_dir)
    tables = {name: _profile_table(name, df) for name, df in frames.items()}
    relations = _detect_relations(tables, frames)
    return ScanReport(tables=tables, relations=relations)


def report_to_dict(report: ScanReport) -> dict[str, Any]:
    return {
        "tables": {
            tname: {
                "rows": t.rows,
                "primary_key": t.pk,
                "pk_candidates": t.pk_candidates,
                "columns": {
                    cname: {
                        "type": c.type,
                        "null_rate": c.null_rate,
                        "n_unique": c.n_unique,
                        "unique": c.unique,
                        "suggestion": c.suggestion,
                    }
                    for cname, c in t.columns.items()
                },
            }
            for tname, t in report.tables.items()
        },
        "relations": [
            {
                "child": r.child,
                "child_column": r.child_column,
                "parent": r.parent,
                "parent_key": r.parent_key,
                "coverage": r.coverage,
                "cardinality": r.cardinality,
                "child_stride": r.child_stride,
                "mean_children": r.mean_children,
                "max_children": r.max_children,
            }
            for r in report.relations
        ],
    }


def render_report(report: ScanReport) -> str:
    """Human-readable summary for ``verisynth scan``."""
    lines: list[str] = []
    for tname, t in report.tables.items():
        pk = t.pk or "?"
        lines.append(f"{tname}  ({t.rows} rows, pk: {pk})")
        for cname, c in t.columns.items():
            flags = []
            if cname == t.pk:
                flags.append("pk")
            for r in report.relations:
                if r.child == tname and r.child_column == cname:
                    flags.append(f"fk -> {r.parent}.{r.parent_key} ({r.coverage:.0%})")
            suffix = f"  [{', '.join(flags)}]" if flags else ""
            null_part = f", {c.null_rate:.1%} null" if c.null_rate else ""
            lines.append(f"  {cname}: {c.type} ({c.n_unique} distinct{null_part}){suffix}")
    if report.relations:
        lines.append("")
        lines.append("relations:")
        for r in report.relations:
            card = ", ".join(f"{k}={v}" for k, v in r.cardinality.items())
            lines.append(
                f"  {r.parent} 1--N {r.child} via {r.child_column} "
                f"(avg {r.mean_children}, max {r.max_children}; {card}; "
                f"child_stride {r.child_stride})"
            )
    return "\n".join(lines)
