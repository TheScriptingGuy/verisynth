"""Metadata scanner: profile real tables and detect structure.

Reads a directory of ``{table}.parquet`` / ``{table}.csv`` files and infers
the structural facts a metadata skeleton needs: column types and null rates,
primary-key candidates, foreign-key relations (with value-containment
evidence), parent/child cardinality profiles, and per-column distribution
suggestions. The interactive wizard (``verisynth init``, see wizard.py) uses
the report as chat suggestions; ``verisynth scan`` prints it directly.

Detection is heuristic and advisory: nothing here mutates data, and every
finding carries the evidence (coverage, counts) so a human can veto it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from ._skeleton_infer import assign_source, infer_skeleton, init_from_dir  # noqa: F401

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
    for f in sorted(p.iterdir()):
        if f.suffix == ".parquet":
            frames[f.stem] = pl.read_parquet(f)
        elif f.suffix == ".csv":
            frames[f.stem] = pl.read_csv(f, try_parse_dates=True)
    if not frames:
        raise FileNotFoundError(f"scan: no .parquet or .csv files found in {p}")
    return frames


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
