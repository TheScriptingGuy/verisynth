"""Plain-language Markdown explanation of a metadata document.

See ``docs/ARCHITECTURE.md`` §2-§5, §8 for the DSL this module renders in
human-readable prose. This module never mutates or validates ``Metadata``;
it is purely a read-only renderer and must not raise on any ``Metadata``
instance (even one bypassing ``validate()`` with an unrecognized
distribution kind — such cases fall back to a generic ``kind(params)``
rendering).
"""

from __future__ import annotations

import math
from typing import Any

from .metadata import DistributionSpec, Metadata, TableSpec
from .temporal import order_temporal_columns

# --------------------------------------------------------------------------
# Small numeric-formatting helpers
# --------------------------------------------------------------------------


def _num(x: Any) -> str:
    """Format a number compactly: integers plain, floats to ~4 significant
    figures with no trailing zeros."""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isfinite(xf) and xf == int(xf):
        return str(int(xf))
    return f"{xf:.4g}"


def _sig(x: float, n: int = 2) -> str:
    """Round ``x`` to ``n`` significant figures, formatted compactly."""
    if x == 0:
        return "0"
    d = n - 1 - int(math.floor(math.log10(abs(x))))
    r = round(x, d)
    if r == int(r):
        return str(int(r))
    return str(r)


def humanize_seconds(s: float) -> str:
    """Render a duration in seconds as a short human phrase.

    ``< 90s`` -> "Xs"; ``< 90 min`` -> "X.Y min"; ``< 36 h`` -> "X.Y h";
    else "X.Y days" (one decimal, trailing ``.0`` stripped).
    """
    s = float(s)

    def _strip(x: float) -> str:
        r = round(x, 1)
        if r == int(r):
            return str(int(r))
        return str(r)

    if s < 90:
        return f"{int(round(s))}s"
    minutes = s / 60
    if minutes < 90:
        return f"{_strip(minutes)} min"
    hours = s / 3600
    if hours < 36:
        return f"{_strip(hours)} h"
    days = s / 86400
    return f"{_strip(days)} days"


# --------------------------------------------------------------------------
# Distribution description
# --------------------------------------------------------------------------


def describe_distribution(spec: DistributionSpec) -> str:
    """Plain-language, deterministic description of a distribution spec."""
    kind = spec.kind
    p = spec.params

    try:
        if kind == "categorical":
            categories = p["categories"]
            probs = p["probs"]
            k = len(categories)
            pairs = sorted(zip(categories, probs), key=lambda cp: -cp[1])[:3]
            parts = [f"{cat} ({prob * 100:.1f}%)" for cat, prob in pairs]
            text = "mostly " + ", ".join(parts) + f" ({k} categories)"
            if k > 3:
                text += f" and {k - 3} more"
            return text

        if kind == "normal":
            return f"centered at {_num(p['mean'])} ± {_num(p['std'])}"

        if kind == "lognormal":
            mu, sigma = float(p["mu"]), float(p["sigma"])
            median = math.exp(mu)
            lo = math.exp(mu - sigma)
            hi = math.exp(mu + sigma)
            return (
                f"skewed, median ≈ {_sig(median)}, "
                f"typical range {_sig(lo)} – {_sig(hi)}"
            )

        if kind in ("uniform", "uniform_int"):
            return f"uniform between {_num(p['low'])} and {_num(p['high'])}"

        if kind == "exponential":
            rate = float(p["rate"])
            mean = 1.0 / rate if rate else float("inf")
            return f"exponential, mean ≈ {_num(mean)}"

        if kind == "gamma":
            shape, scale = float(p["shape"]), float(p["scale"])
            mean = shape * scale
            return f"gamma(shape={_num(shape)}, scale={_num(scale)}), mean ≈ {_num(mean)}"

        if kind == "beta":
            a, b = float(p["a"]), float(p["b"])
            mean = a / (a + b) if (a + b) else 0.0
            return f"beta(a={_num(a)}, b={_num(b)}), mean ≈ {_num(mean)}"

        if kind == "datetime_uniform":
            return f"between {p['start']} and {p['end']}"

        if kind == "zipf":
            from scipy.stats import zipfian

            a, n = float(p["a"]), int(p["n"])
            p0 = float(zipfian.pmf(1, a, n)) * 100
            p10 = float(zipfian.cdf(10, a, n)) * 100
            return (
                f"popularity-ranked over {n} items; most popular ≈ "
                f"{_sig(p0)}% of picks, top 10 ≈ {_sig(p10)}%"
            )
    except Exception:
        pass

    return f"{kind}({p})"


# --------------------------------------------------------------------------
# Cardinality / temporal-delay wording
# --------------------------------------------------------------------------


def _describe_cardinality(child: str, parent: str, card) -> str:
    kind = card.kind
    p = card.params
    try:
        if kind == "bernoulli":
            prob = float(p["p"])
            return f"each {parent} row has one {child} row with probability {prob:.0%}"
        if kind == "poisson":
            lam = float(p["lam"])
            return f"on average {lam:.2g} {child} rows per {parent} row (up to {p['max']})"
        if kind == "fixed":
            return f"exactly {p['n']} per {parent} row"
        if kind == "uniform_int":
            return f"between {p['low']} and {p['high']} per {parent} row"
    except Exception:
        pass
    return f"{kind}({p})"


def _describe_delay(delay: DistributionSpec) -> str:
    kind = delay.kind
    p = delay.params
    try:
        if kind == "lognormal":
            mu, sigma = float(p["mu"]), float(p["sigma"])
            typical = humanize_seconds(math.exp(mu))
            upper = humanize_seconds(math.exp(mu + 2 * sigma))
            return f"typically {typical} (up to ~{upper})"
        if kind == "exponential":
            rate = float(p["rate"])
            mean = 1.0 / rate if rate else float("inf")
            return f"on average {humanize_seconds(mean)}"
    except Exception:
        pass
    return describe_distribution(delay)


# --------------------------------------------------------------------------
# Column description
# --------------------------------------------------------------------------


def _describe_column(table: TableSpec, cname: str) -> str:
    col = table.columns[cname]

    if col.generator is not None:
        if col.generator == "key":
            return "primary key"
        if col.generator == "parent_key":
            return f"reference to the `{table.parent}` row"
        if col.generator.startswith("parent:"):
            parent_col = col.generator[len("parent:"):]
            return (
                f"inherited from `{table.parent}.{parent_col}` "
                "(master data — always identical to the parent's value)"
            )
        return f"generator: {col.generator}"

    if col.document is not None:
        doc = col.document
        kind_name = "JSON object" if doc.kind == "json" else "XML fragment"
        embedded = doc.columns or [
            n for n, cc in table.columns.items() if n != cname and cc.document is None
        ]
        names = ", ".join(f"`{n}`" for n in embedded)
        desc = f"a {kind_name} rendered per row from {names}"
        if doc.schemas:
            desc += " (shaped by " + ", ".join(f"`{s}`" for s in doc.schemas) + ")"
        desc += " — always consistent with those columns"
        if col.null_rate:
            desc += f" ; {col.null_rate:.1%} null"
        return desc

    if col.reference is not None:
        desc = describe_distribution(col.distribution) if col.distribution else ""
        return f"reference into `{col.reference}` — {desc}"

    if col.distribution is not None:
        desc = describe_distribution(col.distribution)
        if col.null_rate:
            desc += f" ; {col.null_rate:.1%} null"
        if col.clamp is not None:
            desc += f" ; clamped to [{_num(col.clamp[0])}, {_num(col.clamp[1])}]"
        return desc

    if col.temporal is not None:
        desc = f"happens {_describe_delay(col.temporal.delay)} after `{col.temporal.anchor}`"
        if col.null_rate:
            desc += f" ; {col.null_rate:.1%} null"
        return desc

    return "(unspecified)"


# --------------------------------------------------------------------------
# Temporal event-flow chains
# --------------------------------------------------------------------------


def _temporal_chains(table: TableSpec) -> list[str]:
    try:
        order = order_temporal_columns(table)
    except Exception:
        return []
    if not order:
        return []

    temporal_set = set(order)
    children: dict[str, list[str]] = {}
    roots: list[str] = []
    root_label: dict[str, str] = {}

    for cname in order:
        anchor = table.columns[cname].temporal.anchor
        if "." in anchor:
            root_key = anchor
            label = anchor
        elif anchor in temporal_set:
            root_key = anchor
            label = None
        else:
            root_key = anchor
            label = anchor
        children.setdefault(root_key, []).append(cname)
        if label is not None and root_key not in root_label:
            root_label[root_key] = label
            roots.append(root_key)

    paths: list[list[str]] = []

    def walk(node: str, path: list[str]) -> None:
        kids = children.get(node, [])
        if not kids:
            paths.append(path)
            return
        for k in kids:
            walk(k, path + [k])

    for root in roots:
        walk(root, [root_label[root]])

    return [
        "Event flow: " + " → ".join(f"`{n}`" for n in path) for path in paths
    ]


# --------------------------------------------------------------------------
# Top-level rendering
# --------------------------------------------------------------------------


def _render_table(md: Metadata, tname: str) -> list[str]:
    t = md.tables[tname]
    lines: list[str] = [f"### {tname}", ""]

    if t.role == "root":
        lines.append(f"Root entity, {t.rows:,} rows.")
    else:
        phrase = _describe_cardinality(tname, t.parent, t.cardinality)
        lines.append(f"Child of `{t.parent}`: {phrase}.")
    lines.append("")

    if t.format is not None:
        kind_desc = {
            "json": "a JSON document file",
            "jsonl": "a JSON Lines document file",
            "xml": "an XML document file",
        }.get(t.format.kind, f"{t.format.kind} documents")
        sentence = f"Also rendered as {kind_desc}"
        if t.format.schemas:
            names = ", ".join(f"`{s}`" for s in t.format.schemas)
            sentence += f" shaped by {names}"
        if t.format.nest:
            nested = ", ".join(
                f"`{n.table}` as `{n.alias or n.table}`" for n in t.format.nest
            )
            sentence += f", nesting {nested} inside each record"
        lines.append(sentence + ".")
        lines.append("")

    for cname, col in t.columns.items():
        lines.append(f"- **{cname}** ({col.type}): {_describe_column(t, cname)}")
    lines.append("")

    for cop in t.copulas:
        n = len(cop.columns)
        for i in range(n):
            for j in range(i + 1, n):
                r = cop.correlation[i][j]
                lines.append(
                    f"Correlations ({cop.name}): {cop.columns[i]} ↔ "
                    f"{cop.columns[j]} at r = {r:.2f}"
                )
        lines.append("")

    for chain in _temporal_chains(t):
        lines.append(chain)
    if t.columns and any(c.temporal is not None for c in t.columns.values()):
        lines.append("")

    for der in t.derived:
        lines.append(f"Derived: `{der.name}` = `{der.expr}`")
    if t.derived:
        lines.append("")

    return lines


def explain_metadata(md: Metadata) -> str:
    """Render ``md`` as a plain-language Markdown explanation."""
    order = md.table_order()
    n_tables = len(order)

    named_sources: list[str] = []
    seen_sources: set[str] = set()
    has_unassigned = False
    for tname in order:
        src = md.tables[tname].source
        if src is None:
            has_unassigned = True
        elif src not in seen_sources:
            seen_sources.add(src)
            named_sources.append(src)

    source_parts = []
    for src in named_sources:
        k = sum(1 for tname in order if md.tables[tname].source == src)
        source_parts.append(f"{src} ({k} table{'s' if k != 1 else ''})")
    if has_unassigned:
        source_parts.append("unassigned")
    sources_desc = ", ".join(source_parts) if source_parts else "unassigned"

    lines: list[str] = [
        f"# Synthetic dataset: {n_tables} tables across {sources_desc}",
        "",
        f"Deterministic generation with seed {md.seed}: identical output for "
        "identical metadata, any partition count.",
        "",
    ]

    # Buckets: source name (first-appearance order) or None for unassigned.
    buckets: list[str | None] = []
    seen_bucket: set[Any] = set()
    for tname in order:
        src = md.tables[tname].source
        if src not in seen_bucket:
            seen_bucket.add(src)
            buckets.append(src)

    no_sources_at_all = not named_sources

    for bucket in buckets:
        if bucket is None:
            header = "## Tables" if no_sources_at_all else "## Source: (unassigned)"
        else:
            header = f"## Source: {bucket}"
        lines.append(header)
        lines.append("")

        for tname in order:
            if md.tables[tname].source != bucket:
                continue
            lines.extend(_render_table(md, tname))

    lines.append("## Privacy")
    lines.append("")
    lines.append(
        "This document contains only fitted statistical parameters "
        "(distribution shapes, correlations, cardinalities, temporal "
        "delays) — no source records, row-level values, or "
        "identifiers are stored in the metadata itself."
    )
    lines.append(
        "This document could be regenerated with differential privacy by "
        "re-running `verisynth fit --epsilon <budget>` against the source "
        "data, which perturbs each released statistic with Laplace noise "
        "before it is written here."
    )

    # Collapse consecutive blank lines and strip trailing whitespace per line.
    cleaned: list[str] = []
    prev_blank = False
    for line in lines:
        line = line.rstrip()
        if line == "":
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False
        cleaned.append(line)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()

    return "\n".join(cleaned) + "\n"
