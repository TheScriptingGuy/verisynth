"""Fit metadata parameters from real data, optionally under differential privacy.

See docs/ARCHITECTURE.md §7 (normative). ``fit_metadata`` never mutates its
``skeleton`` argument: the skeleton is deep-copied up front and only the copy
is filled in and returned.

Differential privacy (``epsilon`` set) release order is fixed for
determinism: tables in ``skeleton.table_order()``, within each table columns
in declaration order (marginals), then cardinality, then copulas (pairs in
``(i, j)`` order with ``i < j``), then temporal delays. Skipped statistics
(e.g. a ``datetime_uniform`` marginal) consume no epsilon budget and no RNG
draws. DP noise is drawn from its own ``numpy.random.default_rng(dp_seed)`` —
it is independent of the deterministic keyed generation RNG.

v0 limitation: ``datetime_uniform`` marginal bounds and temporal delay
parameters are released WITHOUT DP noise. Only numeric/categorical marginals,
child cardinality ``lam``, and copula correlations are perturbed.
"""

from __future__ import annotations

import copy
import math
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np
import polars as pl
from scipy import stats

from .metadata import CardinalitySpec, DistributionSpec, Metadata
from .privacy import laplace_value, noisy_counts, split_epsilon

try:  # pragma: no cover - exercised indirectly via _to_polars
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None

_EPOCH = datetime(1970, 1, 1)

# A DP release entry: (k-cost toward the epsilon split, apply(eps_i, rng)).
_DPEntry = tuple[int, Callable[[float, np.random.Generator], None]]


def fit_metadata(
    frames: dict[str, Any],
    skeleton: Metadata,
    epsilon: float | None = None,
    dp_seed: int = 0,
) -> Metadata:
    """Fit ``skeleton``'s parameters from real ``frames``.

    ``frames`` maps table name -> polars DataFrame or pyarrow Table. Returns
    a new ``Metadata`` (deep copy of ``skeleton``, filled in); ``skeleton``
    itself is never mutated.
    """
    if epsilon is not None and epsilon <= 0:
        raise ValueError(f"epsilon must be > 0 if set (got {epsilon!r})")

    md = copy.deepcopy(skeleton)
    pl_frames = {name: _to_polars(f) for name, f in frames.items()}

    dp_entries: list[_DPEntry] = []

    for tname in md.table_order():
        t = md.tables[tname]
        df = pl_frames.get(tname)
        if df is None:
            continue

        # 1. Marginal columns, in declaration order.
        for cname, col in t.columns.items():
            if col.distribution is None:
                continue
            dist, meta = _fit_marginal(df[cname], col.distribution.kind)
            col.distribution = dist
            if epsilon is not None:
                entry = _marginal_dp_entry(tname, cname, col, dist, meta)
                if entry is not None:
                    dp_entries.append(entry)

        # 2. Cardinality (child tables only).
        if t.role == "child":
            observed_max, n_parents = _fit_cardinality(md, t, pl_frames)
            if epsilon is not None:
                dp_entries.append(_cardinality_dp_entry(t, observed_max, n_parents))

        # 3. Copulas.
        for cop in t.copulas:
            pair_info = _fit_copula(cop, df)
            if epsilon is not None:
                for i, j, n in pair_info:
                    dp_entries.append(_copula_pair_dp_entry(cop, i, j, n))

        # 4. Temporal delays, in declaration order. Never DP-perturbed.
        for cname, col in t.columns.items():
            if col.temporal is None:
                continue
            _fit_temporal(md, t, cname, col, df, pl_frames)

    if epsilon is not None:
        k = sum(cost for cost, _ in dp_entries)
        eps_i = split_epsilon(epsilon, max(k, 1))
        rng = np.random.default_rng(dp_seed)
        for _, apply_fn in dp_entries:
            apply_fn(eps_i, rng)

    return md


def _to_polars(df: Any) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    if pa is not None and isinstance(df, pa.Table):
        return pl.from_arrow(df)
    raise TypeError(f"unsupported frame type: {type(df)!r}")


# --------------------------------------------------------------------------
# 1. Marginal column fitting
# --------------------------------------------------------------------------


def _epoch_seconds_to_iso(seconds: int) -> str:
    return (_EPOCH + timedelta(seconds=int(seconds))).isoformat()


def _fit_datetime(s: pl.Series) -> DistributionSpec:
    us = s.cast(pl.Datetime("us")).to_physical().to_numpy().astype(np.int64)
    if us.size == 0:
        start_s, end_s = 0, 1
    else:
        start_s = math.floor(int(us.min()) / 1_000_000)
        end_s = math.ceil(int(us.max()) / 1_000_000)
        if end_s <= start_s:
            end_s = start_s + 1
    return DistributionSpec(
        kind="datetime_uniform",
        params={"start": _epoch_seconds_to_iso(start_s), "end": _epoch_seconds_to_iso(end_s)},
    )


def _fit_numeric(s: pl.Series) -> tuple[DistributionSpec, dict[str, Any]]:
    x = s.to_numpy().astype(np.float64)
    n = x.size
    if n == 0:
        return (
            DistributionSpec(kind="normal", params={"mean": 0.0, "std": 1e-9}),
            {"kind": "numeric", "n": 0},
        )

    if float(np.min(x)) > 0 and float(stats.skew(x)) > 1.0:
        logx = np.log(x)
        mu = float(np.mean(logx))
        sigma = max(float(np.std(logx, ddof=0)), 1e-9)
        dist = DistributionSpec(kind="lognormal", params={"mu": mu, "sigma": sigma})
    else:
        mean = float(np.mean(x))
        std = max(float(np.std(x, ddof=0)), 1e-9)
        dist = DistributionSpec(kind="normal", params={"mean": mean, "std": std})

    return dist, {"kind": "numeric", "n": n}


def _fit_categorical(s: pl.Series) -> tuple[DistributionSpec, dict[str, Any]]:
    counts_map: dict[Any, int] = {}
    for v in s.to_list():
        counts_map[v] = counts_map.get(v, 0) + 1

    categories = sorted(counts_map.keys())
    counts = np.array([counts_map[c] for c in categories], dtype=np.float64)
    total = float(counts.sum())
    if total > 0:
        probs = [float(c / total) for c in counts]
    else:
        probs = []

    dist = DistributionSpec(
        kind="categorical", params={"categories": categories, "probs": probs}
    )
    return dist, {"kind": "categorical", "counts": counts}


def _fit_marginal(
    series: pl.Series, declared_kind: str | None = None
) -> tuple[DistributionSpec, dict[str, Any]]:
    dtype = series.dtype
    s = series.drop_nulls()

    # If the skeleton declares this column categorical, honor that regardless
    # of dtype (e.g. int64 Likert/enum-style columns). See
    # docs/ARCHITECTURE.md §7 and TASK CARD 9 §1.
    if declared_kind == "categorical":
        return _fit_categorical(s)
    if dtype.is_temporal():
        return _fit_datetime(s), {"kind": "datetime"}
    if dtype.is_numeric():
        return _fit_numeric(s)
    return _fit_categorical(s)


def _marginal_dp_entry(
    tname: str, cname: str, col: Any, dist: DistributionSpec, meta: dict[str, Any]
) -> _DPEntry | None:
    kind = meta["kind"]

    if kind == "numeric":
        if col.clamp is None:
            raise ValueError(
                f"fit_metadata: differential privacy requires 'clamp' on numeric "
                f"column '{tname}.{cname}'"
            )
        lo, hi = col.clamp
        if dist.kind == "lognormal":
            if lo <= 0:
                raise ValueError(
                    f"fit_metadata: differential privacy for lognormal column "
                    f"'{tname}.{cname}' requires clamp lo > 0 (got {lo!r})"
                )
            value_range = math.log(hi) - math.log(lo)
            p_loc, p_scale = "mu", "sigma"
        else:
            value_range = hi - lo
            p_loc, p_scale = "mean", "std"

        n = max(int(meta["n"]), 1)
        sensitivity = value_range / n

        def apply(
            eps_i: float,
            rng: np.random.Generator,
            dist: DistributionSpec = dist,
            p_loc: str = p_loc,
            p_scale: str = p_scale,
            sensitivity: float = sensitivity,
        ) -> None:
            dist.params[p_loc] = float(
                laplace_value(dist.params[p_loc], sensitivity, eps_i, rng)
            )
            scale = laplace_value(dist.params[p_scale], sensitivity, eps_i, rng)
            dist.params[p_scale] = max(float(scale), 1e-9)

        return (2, apply)

    if kind == "categorical":
        counts = meta["counts"]
        n_cat = len(counts)
        if n_cat == 0:
            return None

        def apply(
            eps_i: float,
            rng: np.random.Generator,
            dist: DistributionSpec = dist,
            counts: np.ndarray = counts,
        ) -> None:
            noisy = noisy_counts(counts, eps_i, rng)
            total = max(float(np.sum(noisy)), 1e-9)
            dist.params["probs"] = [float(x / total) for x in noisy]

        return (n_cat, apply)

    return None  # datetime marginals: released without noise, no budget cost


# --------------------------------------------------------------------------
# 2. Cardinality fitting
# --------------------------------------------------------------------------


def _fit_cardinality(md: Metadata, t: Any, pl_frames: dict[str, pl.DataFrame]) -> tuple[float, int]:
    parent_df = pl_frames[t.parent]
    child_df = pl_frames[t.name]
    parent_pk = md.tables[t.parent].primary_key
    fk_col = next(cn for cn, c in t.columns.items() if c.generator == "parent_key")

    counts_df = child_df.group_by(fk_col).agg(pl.len().alias("__count__"))
    joined = (
        parent_df.select(parent_pk)
        .join(
            counts_df,
            left_on=parent_pk,
            right_on=fk_col,
            how="left",
            maintain_order="left",
        )
        .with_columns(pl.col("__count__").fill_null(0))
    )
    counts = joined["__count__"].to_numpy().astype(np.float64)
    n_parents = counts.shape[0]
    lam = float(np.mean(counts)) if n_parents else 0.0
    observed_max = float(np.max(counts)) if n_parents else 0.0
    eff_max = max(1, math.ceil(observed_max * 1.5))

    stride = 1
    while stride <= eff_max:
        stride *= 2

    t.cardinality = CardinalitySpec(
        kind="poisson", params={"lam": max(lam, 1e-9), "max": int(eff_max)}
    )
    t.child_stride = stride

    return observed_max, n_parents


def _cardinality_dp_entry(t: Any, observed_max: float, n_parents: int) -> _DPEntry:
    sensitivity = observed_max / max(n_parents, 1)

    def apply(eps_i: float, rng: np.random.Generator, t: Any = t, sensitivity: float = sensitivity) -> None:
        lam = laplace_value(t.cardinality.params["lam"], sensitivity, eps_i, rng)
        t.cardinality.params["lam"] = max(float(lam), 1e-9)

    return (1, apply)


# --------------------------------------------------------------------------
# 3. Copula fitting
# --------------------------------------------------------------------------


def _fit_copula(cop: Any, df: pl.DataFrame) -> list[tuple[int, int, int]]:
    cols = cop.columns
    k = len(cols)
    R = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    pair_info: list[tuple[int, int, int]] = []

    for i in range(k):
        for j in range(i + 1, k):
            sub = df.select([cols[i], cols[j]]).drop_nulls()
            n = sub.height
            if n < 2:
                rho = 0.0
            else:
                xi = sub[cols[i]].to_numpy().astype(np.float64)
                xj = sub[cols[j]].to_numpy().astype(np.float64)
                rho, _p = stats.spearmanr(xi, xj)
                if not np.isfinite(rho):
                    rho = 0.0
            r = float(np.clip(2.0 * math.sin(math.pi * rho / 6.0), -1.0, 1.0))
            R[i][j] = r
            R[j][i] = r
            pair_info.append((i, j, max(n, 1)))

    cop.correlation = R
    return pair_info


def _copula_pair_dp_entry(cop: Any, i: int, j: int, n: int) -> _DPEntry:
    sensitivity = 6.0 / max(n, 1)

    def apply(eps_i: float, rng: np.random.Generator, cop: Any = cop, i: int = i, j: int = j, sensitivity: float = sensitivity) -> None:
        r = laplace_value(cop.correlation[i][j], sensitivity, eps_i, rng)
        r = float(np.clip(r, -0.99, 0.99))
        cop.correlation[i][j] = r
        cop.correlation[j][i] = r

    return (1, apply)


# --------------------------------------------------------------------------
# 4. Temporal delay fitting (never DP-perturbed)
# --------------------------------------------------------------------------


def _fit_temporal(
    md: Metadata,
    t: Any,
    cname: str,
    col: Any,
    df: pl.DataFrame,
    pl_frames: dict[str, pl.DataFrame],
) -> None:
    anchor_ref = col.temporal.anchor
    event_series = df[cname]
    event_us = event_series.cast(pl.Datetime("us")).to_physical().to_numpy().astype(np.float64)
    event_null = event_series.is_null().to_numpy()

    if "." in anchor_ref:
        ptable, pcol = anchor_ref.split(".", 1)
        parent_df = pl_frames[ptable]
        parent_pk = md.tables[ptable].primary_key
        fk_col = next(cn for cn, c in t.columns.items() if c.generator == "parent_key")
        joined = df.select([fk_col]).join(
            parent_df.select([parent_pk, pcol]),
            left_on=fk_col,
            right_on=parent_pk,
            how="left",
            maintain_order="left",
        )
        anchor_series = joined[pcol]
    else:
        anchor_series = df[anchor_ref]

    anchor_us = anchor_series.cast(pl.Datetime("us")).to_physical().to_numpy().astype(np.float64)
    anchor_null = anchor_series.is_null().to_numpy()

    valid = (~event_null) & (~anchor_null)
    d_seconds = (event_us[valid] - anchor_us[valid]) / 1e6
    d_seconds = d_seconds[np.isfinite(d_seconds)]
    d_seconds = np.maximum(d_seconds, 0.0)

    frac_positive = float(np.mean(d_seconds > 0)) if d_seconds.size > 0 else 0.0

    if d_seconds.size > 0 and frac_positive >= 0.95:
        pos_d = d_seconds[d_seconds > 0]
        logd = np.log(pos_d)
        mu = float(np.median(logd))
        sigma = max(float(np.std(logd, ddof=0)), 1e-9)
        col.temporal.delay = DistributionSpec(kind="lognormal", params={"mu": mu, "sigma": sigma})
    else:
        mean_d = float(np.mean(d_seconds)) if d_seconds.size > 0 else 1e-9
        rate = max(1.0 / max(mean_d, 1e-9), 1e-9)
        col.temporal.delay = DistributionSpec(kind="exponential", params={"rate": rate})
