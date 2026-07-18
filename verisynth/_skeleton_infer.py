"""Deterministic metadata-skeleton inference from real ``{table}.parquet`` data.

Implements TASK CARD 16's fixed inference rules (roles, parents, primary
keys, references, column specs, temporal anchors, copula proposals) plus the
PARENT-AS-PK delta rule agreed with the coordinator. See
docs/ARCHITECTURE.md §2, §3, §7 for the metadata DSL these rules target.

Owned/imported by ``verisynth/scanner.py`` so the interactive chat wizard
(``verisynth init``) and the non-interactive path (``verisynth init --yes``)
share one inference algorithm; the wizard's own advisory ``ScanReport`` /
``Relation`` machinery (used to phrase chat questions) is untouched beyond
the coverage threshold and name-matching simplification described in
scanner.py.

All inference reads at most the first ``SAMPLE_ROWS`` rows per table; every
rule below is a deterministic function of that sample (no randomness --
``seed`` only ends up in the returned ``Metadata.seed``).
"""

from __future__ import annotations

import fnmatch
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy import stats

from .metadata import (
    CardinalitySpec,
    ColumnSpec,
    CopulaSpec,
    DistributionSpec,
    Metadata,
    TableSpec,
    TemporalSpec,
    metadata_to_dict,
    parse_metadata,
)

SAMPLE_ROWS = 100_000
FK_COVERAGE = 0.98
STRING_CATEGORICAL_MAX = 200
STRING_CATEGORICAL_TOP = 20
INT_CATEGORICAL_MAX = 20
COPULA_RHO = 0.3

_EPOCH = datetime(1970, 1, 1)
_PLACEHOLDER_LOGNORMAL = {"mu": 0.0, "sigma": 1.0}


def assign_source(name: str, sources: list[tuple[str, str]] | None) -> str | None:
    """First matching ``(name, fnmatch_pattern)`` pair wins; no match -> None."""
    if not sources:
        return None
    for src_name, pattern in sources:
        if fnmatch.fnmatch(name, pattern):
            return src_name
    return None


def _epoch_seconds_to_iso(seconds: int) -> str:
    return (_EPOCH + timedelta(seconds=int(seconds))).isoformat()


def _dsl_type(dtype: pl.DataType) -> str:
    if dtype == pl.Boolean:
        return "bool"
    if dtype.is_temporal():
        return "timestamp"
    if dtype.is_integer():
        return "int64"
    if dtype.is_float():
        return "float64"
    return "string"


def _containment(child_vals: pl.Series, parent_vals: pl.Series) -> float:
    """Fraction of non-null ``child_vals`` present in ``parent_vals``."""
    nn = child_vals.drop_nulls()
    if nn.len() == 0:
        return 0.0
    parent_unique = parent_vals.drop_nulls().unique()
    if nn.dtype != parent_unique.dtype:
        try:
            nn = nn.cast(parent_unique.dtype)
        except pl.exceptions.PolarsError:
            return 0.0
    return float(nn.is_in(parent_unique.implode()).mean())


def _to_epoch_seconds(series: pl.Series) -> np.ndarray:
    return series.cast(pl.Datetime("us")).to_physical().to_numpy().astype(np.float64) / 1e6


# --------------------------------------------------------------------------
# 1 & delta: primary keys, and the parent-as-pk collision rule
# --------------------------------------------------------------------------


def _unique_candidate_columns(sample: pl.DataFrame) -> list[str]:
    """All-unique, non-null, non-timestamp columns, in schema (declaration)
    order. Timestamp columns are excluded: an incidentally-unique event
    timestamp (fine-grained, e.g. microsecond-resolution ``created_at``) is
    never a meaningful identity column, and every PK is re-typed to int64
    regardless of dtype anyway -- so it is never a sensible fallback PK."""
    h = sample.height
    if h == 0:
        return []
    out = []
    for c in sample.columns:
        col = sample[c]
        if col.dtype.is_temporal():
            continue
        if col.null_count() == 0 and col.n_unique() == h:
            out.append(c)
    return out


def _pick_pk(candidates: list[str]) -> str | None:
    """Rule 1: prefer names ending ``_id``/``id`` (alphabetical tiebreak),
    else the first candidate in schema order. ``None`` if no candidates."""
    if not candidates:
        return None
    id_like = sorted(c for c in candidates if c == "id" or c.endswith("_id"))
    return id_like[0] if id_like else candidates[0]


def _resolve_parent_as_pk(
    tables: list[str],
    samples: dict[str, pl.DataFrame],
    full_rows: dict[str, int],
    unique_cols: dict[str, list[str]],
    warnings: list[str],
) -> dict[str, tuple[str, str, float]]:
    """Detect tables whose only-viable PK column is itself a foreign key into
    another table's PK (name collision + >=98% containment). Returns
    ``child -> (parent, via_column, coverage)``.
    """
    by_name: dict[str, list[str]] = defaultdict(list)
    for t in tables:
        for c in unique_cols[t]:
            by_name[c].append(t)

    forced: dict[str, tuple[str, str, float]] = {}
    for name in sorted(by_name):
        members = by_name[name]
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                t1, t2 = members[i], members[j]
                v1, v2 = samples[t1][name], samples[t2][name]
                cov_1in2 = _containment(v1, v2)
                cov_2in1 = _containment(v2, v1)
                q1, q2 = cov_1in2 >= FK_COVERAGE, cov_2in1 >= FK_COVERAGE
                if not q1 and not q2:
                    continue
                if q1 and q2:
                    if full_rows[t1] != full_rows[t2]:
                        parent = t1 if full_rows[t1] > full_rows[t2] else t2
                    else:
                        parent = min(t1, t2)
                    child = t2 if parent == t1 else t1
                    coverage = cov_1in2 if child == t1 else cov_2in1
                elif q1:
                    parent, child, coverage = t2, t1, cov_1in2
                else:
                    parent, child, coverage = t1, t2, cov_2in1

                if child in forced:
                    warnings.append(
                        f"{child}: key column {name!r} also collides with "
                        f"{parent!r}'s key; keeping the first match "
                        f"({forced[child][0]!r})"
                    )
                    continue
                forced[child] = (parent, name, coverage)
    return forced


def _finalize_primary_keys(
    tables: list[str],
    unique_cols: dict[str, list[str]],
    forced_parent: dict[str, tuple[str, str, float]],
    warnings: list[str],
) -> dict[str, str]:
    final_pk: dict[str, str] = {}
    for t in tables:
        claimed = forced_parent[t][1] if t in forced_parent else None
        candidates = [c for c in unique_cols[t] if c != claimed]
        pk = _pick_pk(candidates)
        if pk is None:
            pk = f"{t}_id"
            if t in forced_parent:
                warnings.append(
                    f"{t}: primary-key column {forced_parent[t][1]!r} is itself a "
                    f"foreign key into {forced_parent[t][0]!r} (parent-as-pk rule); "
                    f"minted synthetic primary key {pk!r}"
                )
            else:
                warnings.append(
                    f"{t}: no unique non-null column found; minted synthetic "
                    f"primary key {pk!r}"
                )
        final_pk[t] = pk
    return final_pk


# --------------------------------------------------------------------------
# 2 & 3: FK candidates -> parent vs reference
# --------------------------------------------------------------------------


def _resolve_parents_and_references(
    tables: list[str],
    samples: dict[str, pl.DataFrame],
    final_pk: dict[str, str],
    forced_parent: dict[str, tuple[str, str, float]],
) -> tuple[dict[str, str | None], dict[str, str | None], dict[str, float], dict[str, list[tuple[str, str]]]]:
    parent_of: dict[str, str | None] = {}
    fk_col_of: dict[str, str | None] = {}
    coverage_of: dict[str, float] = {}
    reference_of: dict[str, list[tuple[str, str]]] = {t: [] for t in tables}

    for t in tables:
        s = samples[t]
        own_pk = final_pk[t]
        excluded = {own_pk}
        if t in forced_parent:
            excluded.add(forced_parent[t][1])

        candidates: list[tuple[str, str, float, float]] = []
        for c in s.columns:
            if c in excluded:
                continue
            for p in tables:
                if p == t or final_pk[p] != c:
                    continue
                coverage = _containment(s[c], samples[p][final_pk[p]])
                if coverage < FK_COVERAGE:
                    continue
                distinct_c = s[c].drop_nulls().n_unique()
                mean_children = (s.height / distinct_c) if distinct_c > 0 else float("inf")
                candidates.append((c, p, coverage, mean_children))

        if t in forced_parent:
            parent_of[t] = forced_parent[t][0]
            fk_col_of[t] = forced_parent[t][1]
            coverage_of[t] = forced_parent[t][2]
            reference_of[t] = [(c, p) for c, p, _cov, _mc in candidates]
        elif candidates:
            best = min(candidates, key=lambda x: (x[3], x[1]))
            parent_of[t] = best[1]
            fk_col_of[t] = best[0]
            coverage_of[t] = best[2]
            reference_of[t] = [
                (c, p) for c, p, _cov, _mc in candidates if (c, p) != (best[0], best[1])
            ]
        else:
            parent_of[t] = None
            fk_col_of[t] = None

    return parent_of, fk_col_of, coverage_of, reference_of


def _break_parent_cycles(
    tables: list[str],
    parent_of: dict[str, str | None],
    fk_col_of: dict[str, str | None],
    coverage_of: dict[str, float],
    warnings: list[str],
) -> None:
    """Repeatedly find a cycle in the parent graph and drop the lowest-
    coverage edge in it (that table becomes a root), until none remain."""

    def find_cycle() -> list[str] | None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def dfs(node: str, chain: list[str]) -> list[str] | None:
            if node not in parent_of or parent_of.get(node) is None:
                return None
            if node in visited:
                return None
            if node in visiting:
                return chain[chain.index(node):]
            visiting.add(node)
            result = dfs(parent_of[node], chain + [node])
            visiting.discard(node)
            visited.add(node)
            return result

        for t in tables:
            cyc = dfs(t, [])
            if cyc:
                return cyc
        return None

    while True:
        cyc = find_cycle()
        if not cyc:
            return
        worst = min(cyc, key=lambda x: coverage_of.get(x, 1.0))
        warnings.append(
            f"{worst}: dropped parent link to {parent_of[worst]!r} to break a "
            "cycle in the inferred parent graph"
        )
        parent_of[worst] = None
        fk_col_of[worst] = None


# --------------------------------------------------------------------------
# 4: cardinality
# --------------------------------------------------------------------------


def _infer_cardinality(sample: pl.DataFrame, fk_col: str) -> tuple[CardinalitySpec, int]:
    counts = (
        sample.select(fk_col)
        .drop_nulls()
        .group_by(fk_col)
        .len()["len"]
        .to_numpy()
        .astype(np.float64)
    )
    if counts.size == 0 or bool(np.all(counts <= 1)):
        return CardinalitySpec(kind="bernoulli", params={"p": 0.5}), 2

    mean = float(np.mean(counts))
    observed_max = float(np.max(counts))
    eff_max = max(1, math.ceil(observed_max * 1.5))
    stride = 1
    while stride <= eff_max:
        stride *= 2
    return (
        CardinalitySpec(kind="poisson", params={"lam": round(mean, 2), "max": int(eff_max)}),
        stride,
    )


# --------------------------------------------------------------------------
# 5: plain column specs
# --------------------------------------------------------------------------


def _infer_plain_column(
    tname: str, cname: str, series: pl.Series, warnings: list[str]
) -> tuple[ColumnSpec | None, bool]:
    """Returns ``(spec_or_None, eligible_for_copula)``; ``None`` means the
    column is omitted (high-cardinality string)."""
    dtype = series.dtype
    n = series.len()
    null_frac = (series.null_count() / n) if n else 0.0

    if dtype == pl.Boolean:
        dist = DistributionSpec(kind="categorical", params={"categories": [True, False], "probs": [0.5, 0.5]})
        eligible = False
    elif dtype.is_float():
        dist = DistributionSpec(kind="lognormal", params=dict(_PLACEHOLDER_LOGNORMAL))
        eligible = True
    elif dtype.is_integer():
        nn = series.drop_nulls()
        distinct_vals = sorted({int(x) for x in nn.unique().to_list()})
        if len(distinct_vals) <= INT_CATEGORICAL_MAX:
            k = len(distinct_vals)
            dist = DistributionSpec(
                kind="categorical",
                params={"categories": distinct_vals, "probs": [1.0 / k] * k},
            )
            eligible = True
        else:
            dist = DistributionSpec(kind="lognormal", params=dict(_PLACEHOLDER_LOGNORMAL))
            eligible = True
    else:  # string
        nn = series.drop_nulls()
        distinct = nn.n_unique()
        if distinct > STRING_CATEGORICAL_MAX:
            warnings.append(f"high-cardinality string column {tname}.{cname} omitted")
            return None, False
        vc = nn.value_counts()
        vc = vc.sort(by=["count", cname], descending=[True, False])
        cats = vc[cname].head(STRING_CATEGORICAL_TOP).to_list()
        k = len(cats)
        dist = DistributionSpec(kind="categorical", params={"categories": cats, "probs": [1.0 / k] * k})
        eligible = False

    col = ColumnSpec(name=cname, type=_dsl_type(dtype), distribution=dist)
    if null_frac > 0:
        col.null_rate = round(null_frac, 4)
    return col, eligible


# --------------------------------------------------------------------------
# 6: temporal anchors
# --------------------------------------------------------------------------


def _infer_temporal_columns(
    tname: str,
    role: str,
    parent: str | None,
    fk_col: str | None,
    samples: dict[str, pl.DataFrame],
    final_pk: dict[str, str],
    ts_cols: list[str],
    warnings: list[str],
) -> dict[str, ColumnSpec]:
    s = samples[tname]
    if not ts_cols:
        return {}

    medians: dict[str, float] = {}
    for c in ts_cols:
        nn = s[c].drop_nulls()
        medians[c] = float(np.median(_to_epoch_seconds(nn))) if nn.len() else float("inf")
    order = sorted(ts_cols, key=lambda c: medians[c])

    processed: dict[str, np.ndarray] = {}
    processed_mask: dict[str, np.ndarray] = {}

    if role == "child" and parent is not None:
        parent_sample = samples[parent]
        parent_pk = final_pk[parent]
        fk_vals = s[fk_col].to_frame(fk_col)
        for pc in parent_sample.columns:
            if not parent_sample[pc].dtype.is_temporal():
                continue
            joined = fk_vals.join(
                parent_sample.select([parent_pk, pc]),
                left_on=fk_col,
                right_on=parent_pk,
                how="left",
                maintain_order="left",
            )
            joined_series = joined[pc]
            mask = joined_series.is_not_null().to_numpy()
            vals = np.zeros(joined_series.len(), dtype=np.float64)
            if mask.any():
                vals[mask] = _to_epoch_seconds(joined_series.drop_nulls())
            processed[f"{parent}.{pc}"] = vals
            processed_mask[f"{parent}.{pc}"] = mask

    specs: dict[str, ColumnSpec] = {}
    for c in order:
        b_series = s[c]
        b_mask = b_series.is_not_null().to_numpy()
        b_vals = np.zeros(len(b_series), dtype=np.float64)
        if b_mask.any():
            b_vals[b_mask] = _to_epoch_seconds(b_series.drop_nulls())

        best_name: str | None = None
        best_median: float | None = None
        for cand_name, a_vals in processed.items():
            a_mask = processed_mask[cand_name]
            valid = a_mask & b_mask
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            delta = b_vals[valid] - a_vals[valid]
            frac_nonneg = float(np.mean(delta >= 0))
            med = float(np.median(delta))
            if frac_nonneg >= 0.98 and med > 0:
                if best_median is None or med < best_median:
                    best_name, best_median = cand_name, med

        if best_name is not None:
            delay = DistributionSpec(kind="lognormal", params={"mu": 10.0, "sigma": 1.0})
            spec = ColumnSpec(
                name=c, type="timestamp", temporal=TemporalSpec(anchor=best_name, delay=delay)
            )
        else:
            nn = b_series.drop_nulls()
            if nn.len() == 0:
                start_s, end_s = 0, 1
            else:
                us = nn.cast(pl.Datetime("us")).to_physical().to_numpy().astype(np.int64)
                start_s = math.floor(int(us.min()) / 1_000_000)
                end_s = math.ceil(int(us.max()) / 1_000_000)
                if end_s <= start_s:
                    end_s = start_s + 1
            dist = DistributionSpec(
                kind="datetime_uniform",
                params={"start": _epoch_seconds_to_iso(start_s), "end": _epoch_seconds_to_iso(end_s)},
            )
            spec = ColumnSpec(name=c, type="timestamp", distribution=dist)

        if b_series.len():
            null_frac = b_series.null_count() / b_series.len()
            if null_frac > 0:
                spec.null_rate = round(null_frac, 4)

        specs[c] = spec
        processed[c] = b_vals
        processed_mask[c] = b_mask

    return specs


# --------------------------------------------------------------------------
# 7: copulas
# --------------------------------------------------------------------------


def _infer_copulas(sample: pl.DataFrame, eligible_cols: list[str]) -> list[CopulaSpec]:
    cols = sorted(eligible_cols)
    if len(cols) < 2:
        return []

    parent = {c: c for c in cols}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            sub = sample.select([cols[i], cols[j]]).drop_nulls()
            if sub.height < 2:
                continue
            xi = sub[cols[i]].to_numpy().astype(np.float64)
            xj = sub[cols[j]].to_numpy().astype(np.float64)
            rho, _p = stats.spearmanr(xi, xj)
            if np.isfinite(rho) and abs(rho) >= COPULA_RHO:
                union(cols[i], cols[j])

    groups: dict[str, list[str]] = defaultdict(list)
    for c in cols:
        groups[find(c)].append(c)
    components = [sorted(v) for v in groups.values() if len(v) >= 2]
    components.sort(key=lambda comp: comp[0])

    result = []
    for idx, comp in enumerate(components, start=1):
        k = len(comp)
        corr = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
        result.append(CopulaSpec(name=f"corr_{idx}", columns=comp, correlation=corr))
    return result


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def infer_skeleton(
    frames: dict[str, pl.DataFrame],
    seed: int = 42,
    sources: list[tuple[str, str]] | None = None,
) -> tuple[Metadata, list[str]]:
    """Infer a structurally-valid metadata skeleton from real ``frames``.

    See module docstring / TASK CARD 16 for the rule set. Deterministic:
    calling this twice with the same ``frames`` yields an identical
    ``metadata_to_dict`` output.
    """
    warnings: list[str] = []
    tables = sorted(frames)
    samples = {t: frames[t].head(SAMPLE_ROWS) for t in tables}
    full_rows = {t: frames[t].height for t in tables}

    unique_cols = {t: _unique_candidate_columns(samples[t]) for t in tables}
    forced_parent = _resolve_parent_as_pk(tables, samples, full_rows, unique_cols, warnings)
    final_pk = _finalize_primary_keys(tables, unique_cols, forced_parent, warnings)
    parent_of, fk_col_of, coverage_of, reference_of = _resolve_parents_and_references(
        tables, samples, final_pk, forced_parent
    )
    _break_parent_cycles(tables, parent_of, fk_col_of, coverage_of, warnings)

    string_key_warned: set[str] = set()

    def _warn_string_key(t: str, col_name: str, dtype: pl.DataType) -> None:
        if t in string_key_warned:
            return
        if dtype == pl.Utf8 or dtype == pl.String:
            warnings.append(
                f"{t}: real key column(s) use string ids in the source data; "
                "mapped to synthetic int64 keys"
            )
            string_key_warned.add(t)

    table_specs: dict[str, TableSpec] = {}

    for t in tables:
        s = samples[t]
        role = "child" if parent_of[t] is not None else "root"
        pk_col = final_pk[t]
        fk_col = fk_col_of[t]

        columns: dict[str, ColumnSpec] = {}
        columns[pk_col] = ColumnSpec(name=pk_col, type="int64", generator="key")
        if pk_col in s.columns:
            _warn_string_key(t, pk_col, s[pk_col].dtype)

        if role == "child":
            columns[fk_col] = ColumnSpec(name=fk_col, type="int64", generator="parent_key")
            _warn_string_key(t, fk_col, s[fk_col].dtype)

        handled = {pk_col}
        if fk_col:
            handled.add(fk_col)

        for col_name, ptable in sorted(reference_of[t]):
            null_frac = (s[col_name].null_count() / s.height) if s.height else 0.0
            ref_spec = ColumnSpec(
                name=col_name,
                type="int64",
                distribution=DistributionSpec(kind="zipf", params={"a": 0.5, "n": full_rows[ptable]}),
                reference=ptable,
            )
            if null_frac > 0:
                ref_spec.null_rate = round(null_frac, 4)
            columns[col_name] = ref_spec
            handled.add(col_name)

        ts_cols = [c for c in s.columns if c not in handled and s[c].dtype.is_temporal()]
        handled.update(ts_cols)

        eligible_cols: list[str] = []
        for cname in s.columns:
            if cname in handled:
                continue
            spec, eligible = _infer_plain_column(t, cname, s[cname], warnings)
            if spec is None:
                continue
            columns[cname] = spec
            if eligible:
                eligible_cols.append(cname)

        temporal_specs = _infer_temporal_columns(
            t, role, parent_of[t], fk_col, samples, final_pk, ts_cols, warnings
        )
        columns.update(temporal_specs)

        copulas = _infer_copulas(s, eligible_cols)

        source = assign_source(t, sources)

        if role == "root":
            table_specs[t] = TableSpec(
                name=t,
                role="root",
                columns=columns,
                primary_key=pk_col,
                rows=full_rows[t],
                copulas=copulas,
                source=source,
            )
        else:
            cardinality, child_stride = (
                (CardinalitySpec(kind="bernoulli", params={"p": 0.5}), 2)
                if t in forced_parent
                else _infer_cardinality(s, fk_col)
            )
            table_specs[t] = TableSpec(
                name=t,
                role="child",
                columns=columns,
                primary_key=pk_col,
                parent=parent_of[t],
                cardinality=cardinality,
                child_stride=child_stride,
                copulas=copulas,
                source=source,
            )

    md = Metadata(version=1, seed=seed, tables=table_specs)
    md = parse_metadata(metadata_to_dict(md))
    return md, warnings


def init_from_dir(
    input_dir: str | Path,
    out_path: str | Path,
    seed: int = 42,
    sources: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Load ``{table}.parquet`` files from ``input_dir``, infer a skeleton,
    write it to ``out_path`` as YAML, and return the inference warnings."""
    p = Path(input_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"init: {p} is not a directory")
    frames: dict[str, pl.DataFrame] = {}
    for f in sorted(p.iterdir()):
        if f.suffix == ".parquet":
            frames[f.stem] = pl.read_parquet(f)
    if not frames:
        raise FileNotFoundError(f"init: no .parquet files found in {p}")

    skeleton, warnings = infer_skeleton(frames, seed=seed, sources=sources)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        yaml.safe_dump(metadata_to_dict(skeleton), fh, sort_keys=False)

    return warnings
