"""Metadata DSL: dataclasses + YAML/JSON load & validation.

See docs/ARCHITECTURE.md section 2 for the normative spec this module
implements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


class MetadataError(Exception):
    """Raised when a metadata document fails to parse or validate.

    Message names the offending path, e.g. ``tables.orders.columns.order_total``.
    """


# --------------------------------------------------------------------------
# Dataclasses
# --------------------------------------------------------------------------


@dataclass
class DistributionSpec:
    kind: str
    params: dict[str, Any]


@dataclass
class CardinalitySpec:
    kind: str
    params: dict[str, Any]


@dataclass
class TemporalSpec:
    anchor: str
    delay: DistributionSpec


@dataclass
class DerivedSpec:
    name: str
    expr: str


@dataclass
class CopulaSpec:
    name: str
    columns: list[str]
    correlation: list[list[float]]


@dataclass
class ColumnSpec:
    name: str
    type: str
    generator: str | None = None  # "key" | "parent_key" | "parent:{column}"
    distribution: DistributionSpec | None = None
    temporal: TemporalSpec | None = None
    clamp: tuple[float, float] | None = None
    round: bool = False
    null_rate: float = 0.0
    reference: str | None = None  # dimension reference: name of a root table (§2)


@dataclass
class TableSpec:
    name: str
    role: str  # "root" | "child"
    columns: dict[str, ColumnSpec]
    primary_key: str
    rows: int | None = None  # root only
    parent: str | None = None  # child only
    cardinality: CardinalitySpec | None = None
    child_stride: int | None = None
    copulas: list[CopulaSpec] = field(default_factory=list)
    derived: list[DerivedSpec] = field(default_factory=list)
    source: str | None = None  # optional source-system name (see §6, §8)


@dataclass
class Metadata:
    version: int
    seed: int
    tables: dict[str, TableSpec]

    def table_order(self) -> list[str]:
        """Topological order: roots first, parents before children."""
        order: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str, chain: list[str]) -> None:
            if name in visited:
                return
            if name in visiting:
                raise MetadataError(
                    f"tables.{name}.parent: cycle detected in parent graph: "
                    f"{' -> '.join(chain + [name])}"
                )
            visiting.add(name)
            parent = self.tables[name].parent
            if parent is not None and parent in self.tables:
                visit(parent, chain + [name])
            visiting.discard(name)
            visited.add(name)
            order.append(name)

        for name in self.tables:
            visit(name, [])
        return order


# --------------------------------------------------------------------------
# Known kinds / required params
# --------------------------------------------------------------------------

_COLUMN_TYPES = {"int64", "float64", "string", "bool", "timestamp"}

_DIST_SPECS: dict[str, set[str]] = {
    "categorical": {"categories", "probs"},
    "normal": {"mean", "std"},
    "lognormal": {"mu", "sigma"},
    "uniform": {"low", "high"},
    "exponential": {"rate"},
    "gamma": {"shape", "scale"},
    "beta": {"a", "b"},
    "uniform_int": {"low", "high"},
    "datetime_uniform": {"start", "end"},
    "zipf": {"a", "n"},
}

_DELAY_DIST_KINDS = set(_DIST_SPECS) - {"categorical", "datetime_uniform"}

# Distribution kinds whose ppf() returns an integer-valued (row-index-like)
# sample, and are therefore eligible to back a `reference:` dimension column.
_REFERENCE_DIST_KINDS = {"zipf", "uniform_int"}

_CARD_SPECS: dict[str, set[str]] = {
    "poisson": {"lam", "max"},
    "uniform_int": {"low", "high", "max"},
    "fixed": {"n"},
    "bernoulli": {"p"},
}

# Parameter names that must stay integer-typed rather than being coerced to float.
_DIST_INT_PARAMS: dict[str, set[str]] = {"uniform_int": {"low", "high"}, "zipf": {"n"}}
_CARD_INT_PARAMS: dict[str, set[str]] = {
    "uniform_int": {"low", "high", "max"},
    "poisson": {"max"},
    "fixed": {"n"},
}


# --------------------------------------------------------------------------
# Loading / parsing
# --------------------------------------------------------------------------


def load_metadata(path: str | Path) -> Metadata:
    """Load a metadata document from a ``.yaml``/``.yml``/``.json`` file."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml

        with open(p, "r") as f:
            obj = yaml.safe_load(f)
    elif suffix == ".json":
        with open(p, "r") as f:
            obj = json.load(f)
    else:
        raise MetadataError(
            f"root: unsupported metadata file extension {p.suffix!r} for {p}"
        )
    return parse_metadata(obj)


def parse_metadata(obj: dict) -> Metadata:
    """Parse a plain dict (already-loaded YAML/JSON) into a validated Metadata."""
    if not isinstance(obj, dict):
        raise MetadataError("root: metadata document must be a mapping")

    version = obj.get("version")
    seed = obj.get("seed", 0)

    tables_obj = obj.get("tables")
    if not isinstance(tables_obj, dict) or not tables_obj:
        raise MetadataError("tables: must be a non-empty mapping of table name -> spec")

    tables: dict[str, TableSpec] = {}
    for tname, tdict in tables_obj.items():
        tables[tname] = _parse_table(tname, tdict)

    md = Metadata(version=version, seed=seed, tables=tables)
    validate(md)
    return md


def _coerce(v: Any, int_names: set[str], key: str) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, list):
        return [_coerce(x, int_names, key) for x in v]
    if isinstance(v, (int, float)):
        return int(v) if key in int_names else float(v)
    return v


# Param names holding lists of category *values* (not counts/probabilities):
# preserved exactly as loaded, with no numeric widening/coercion applied.
_NO_COERCE_PARAMS = {"categories"}


def _parse_distribution(path: str, d: Any) -> DistributionSpec:
    if not isinstance(d, dict) or "kind" not in d:
        raise MetadataError(f"{path}: distribution spec must be a mapping with a 'kind'")
    d = dict(d)
    kind = d.pop("kind")
    int_names = _DIST_INT_PARAMS.get(kind, set())
    params = {
        k: (v if k in _NO_COERCE_PARAMS else _coerce(v, int_names, k)) for k, v in d.items()
    }
    return DistributionSpec(kind=kind, params=params)


def _parse_cardinality(path: str, d: Any) -> CardinalitySpec:
    if not isinstance(d, dict) or "kind" not in d:
        raise MetadataError(f"{path}: cardinality spec must be a mapping with a 'kind'")
    d = dict(d)
    kind = d.pop("kind")
    int_names = _CARD_INT_PARAMS.get(kind, set())
    params = {
        k: (v if k in _NO_COERCE_PARAMS else _coerce(v, int_names, k)) for k, v in d.items()
    }
    return CardinalitySpec(kind=kind, params=params)


def _parse_temporal(path: str, d: Any) -> TemporalSpec:
    if not isinstance(d, dict):
        raise MetadataError(f"{path}: temporal spec must be a mapping")
    anchor = d.get("anchor")
    delay_obj = d.get("delay")
    if delay_obj is None:
        raise MetadataError(f"{path}.delay: required")
    delay = _parse_distribution(f"{path}.delay", delay_obj)
    return TemporalSpec(anchor=anchor, delay=delay)


def _parse_copula(path: str, d: Any) -> CopulaSpec:
    if not isinstance(d, dict):
        raise MetadataError(f"{path}: copula spec must be a mapping")
    name = d.get("name")
    columns = list(d.get("columns") or [])
    correlation = [[float(x) for x in row] for row in (d.get("correlation") or [])]
    return CopulaSpec(name=name, columns=columns, correlation=correlation)


def _parse_derived(path: str, d: Any) -> DerivedSpec:
    if not isinstance(d, dict):
        raise MetadataError(f"{path}: derived spec must be a mapping")
    return DerivedSpec(name=d.get("name"), expr=d.get("expr"))


def _parse_column(path: str, cname: str, cdict: Any) -> ColumnSpec:
    if not isinstance(cdict, dict):
        raise MetadataError(f"{path}: column spec must be a mapping")

    distribution = None
    if cdict.get("distribution") is not None:
        distribution = _parse_distribution(f"{path}.distribution", cdict["distribution"])

    temporal = None
    if cdict.get("temporal") is not None:
        temporal = _parse_temporal(f"{path}.temporal", cdict["temporal"])

    clamp = cdict.get("clamp")
    if clamp is not None:
        clamp = tuple(float(x) for x in clamp)

    return ColumnSpec(
        name=cname,
        type=cdict.get("type"),
        generator=cdict.get("generator"),
        distribution=distribution,
        temporal=temporal,
        clamp=clamp,
        round=bool(cdict.get("round", False)),
        null_rate=float(cdict.get("null_rate", 0.0)),
        reference=cdict.get("reference"),
    )


def _parse_table(tname: str, tdict: Any) -> TableSpec:
    path = f"tables.{tname}"
    if not isinstance(tdict, dict):
        raise MetadataError(f"{path}: table spec must be a mapping")

    columns_obj = tdict.get("columns")
    if not isinstance(columns_obj, dict) or not columns_obj:
        raise MetadataError(f"{path}.columns: must be a non-empty mapping")

    columns: dict[str, ColumnSpec] = {}
    for cname, cdict in columns_obj.items():
        columns[cname] = _parse_column(f"{path}.columns.{cname}", cname, cdict)

    cardinality = None
    if tdict.get("cardinality") is not None:
        cardinality = _parse_cardinality(f"{path}.cardinality", tdict["cardinality"])

    copulas = [
        _parse_copula(f"{path}.copulas[{i}]", cop)
        for i, cop in enumerate(tdict.get("copulas") or [])
    ]
    derived = [
        _parse_derived(f"{path}.derived[{i}]", der)
        for i, der in enumerate(tdict.get("derived") or [])
    ]

    return TableSpec(
        name=tname,
        role=tdict.get("role"),
        columns=columns,
        primary_key=tdict.get("primary_key"),
        rows=tdict.get("rows"),
        parent=tdict.get("parent"),
        cardinality=cardinality,
        child_stride=tdict.get("child_stride"),
        copulas=copulas,
        derived=derived,
        source=tdict.get("source"),
    )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _require_positive(path: str, params: dict[str, Any], key: str) -> None:
    v = params.get(key)
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
        raise MetadataError(f"{path}: {key!r} must be > 0 (got {v!r})")


def _validate_distribution(
    path: str, dist: DistributionSpec, allowed_kinds: set[str] | None = None
) -> None:
    if dist.kind not in _DIST_SPECS:
        raise MetadataError(f"{path}: unknown distribution kind {dist.kind!r}")
    if allowed_kinds is not None and dist.kind not in allowed_kinds:
        raise MetadataError(f"{path}: distribution kind {dist.kind!r} is not allowed here")

    required = _DIST_SPECS[dist.kind]
    got = set(dist.params)
    missing = required - got
    extra = got - required
    if missing:
        raise MetadataError(f"{path}: distribution {dist.kind!r} missing params {sorted(missing)}")
    if extra:
        raise MetadataError(f"{path}: distribution {dist.kind!r} has unexpected params {sorted(extra)}")

    p = dist.params
    if dist.kind == "categorical":
        categories = p["categories"]
        probs = p["probs"]
        if not isinstance(categories, list) or not isinstance(probs, list):
            raise MetadataError(f"{path}: categorical categories/probs must be lists")
        if len(categories) != len(probs):
            raise MetadataError(f"{path}: categorical categories and probs must be the same length")
        if any(not isinstance(x, (int, float)) or isinstance(x, bool) or x < 0 for x in probs):
            raise MetadataError(f"{path}: categorical probs must all be >= 0")
        total = sum(probs)
        if abs(total - 1.0) > 1e-6:
            raise MetadataError(f"{path}: categorical probs must sum to 1 (got {total})")
    elif dist.kind == "normal":
        _require_positive(path, p, "std")
    elif dist.kind == "lognormal":
        _require_positive(path, p, "sigma")
    elif dist.kind == "uniform":
        if not (p["low"] < p["high"]):
            raise MetadataError(f"{path}: uniform requires low < high")
    elif dist.kind == "exponential":
        _require_positive(path, p, "rate")
    elif dist.kind == "gamma":
        _require_positive(path, p, "shape")
        _require_positive(path, p, "scale")
    elif dist.kind == "beta":
        _require_positive(path, p, "a")
        _require_positive(path, p, "b")
    elif dist.kind == "uniform_int":
        low, high = p["low"], p["high"]
        if not (isinstance(low, int) and isinstance(high, int)):
            raise MetadataError(f"{path}: uniform_int low/high must be integers")
        if not (low <= high):
            raise MetadataError(f"{path}: uniform_int requires low <= high")
    elif dist.kind == "datetime_uniform":
        start, end = p.get("start"), p.get("end")
        try:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
        except (TypeError, ValueError) as e:
            raise MetadataError(
                f"{path}: datetime_uniform start/end must be ISO-8601 strings ({e})"
            ) from e
        if not (start_dt < end_dt):
            raise MetadataError(f"{path}: datetime_uniform requires start < end")
    elif dist.kind == "zipf":
        a, n = p.get("a"), p.get("n")
        if not isinstance(a, (int, float)) or isinstance(a, bool) or not (a >= 0):
            raise MetadataError(f"{path}: zipf 'a' must be >= 0 (got {a!r})")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise MetadataError(f"{path}: zipf 'n' must be an integer >= 1 (got {n!r})")


def _validate_cardinality(path: str, card: CardinalitySpec, child_stride: int | None) -> None:
    if card.kind not in _CARD_SPECS:
        raise MetadataError(f"{path}: unknown cardinality kind {card.kind!r}")

    required = _CARD_SPECS[card.kind]
    got = set(card.params)
    missing = required - got
    extra = got - required
    if missing:
        raise MetadataError(f"{path}: cardinality {card.kind!r} missing params {sorted(missing)}")
    if extra:
        raise MetadataError(f"{path}: cardinality {card.kind!r} has unexpected params {sorted(extra)}")

    p = card.params
    if card.kind == "poisson":
        _require_positive(path, p, "lam")
        eff_max = p["max"]
    elif card.kind == "uniform_int":
        low, high = p["low"], p["high"]
        if not (isinstance(low, int) and isinstance(high, int)):
            raise MetadataError(f"{path}: uniform_int low/high must be integers")
        if not (low <= high):
            raise MetadataError(f"{path}: uniform_int requires low <= high")
        eff_max = p["max"]
    elif card.kind == "bernoulli":
        p_val = p["p"]
        if not isinstance(p_val, (int, float)) or isinstance(p_val, bool) or not (0.0 <= p_val <= 1.0):
            raise MetadataError(f"{path}: bernoulli 'p' must be within [0, 1] (got {p_val!r})")
        eff_max = 1
    else:  # fixed
        eff_max = p["n"]

    if not isinstance(eff_max, int) or isinstance(eff_max, bool):
        raise MetadataError(f"{path}: effective max must be an integer (got {eff_max!r})")
    if child_stride is None or not (eff_max < child_stride):
        raise MetadataError(
            f"{path}: effective max ({eff_max}) must be < child_stride ({child_stride})"
        )


def _validate_temporal(md: Metadata, tname: str, t: TableSpec) -> None:
    path = f"tables.{tname}"
    temporal_cols = {cname: c.temporal for cname, c in t.columns.items() if c.temporal is not None}

    same_table_graph: dict[str, str] = {}

    for cname, temp in temporal_cols.items():
        cpath = f"{path}.columns.{cname}.temporal"
        anchor = temp.anchor
        if not anchor:
            raise MetadataError(f"{cpath}.anchor: required")

        if "." in anchor:
            ptable, pcol = anchor.split(".", 1)
            if t.role != "child" or ptable != t.parent:
                raise MetadataError(
                    f"{cpath}.anchor: {anchor!r} must reference this table's parent "
                    f"({t.parent!r})"
                )
            parent_spec = md.tables.get(ptable)
            if parent_spec is None or pcol not in parent_spec.columns:
                raise MetadataError(f"{cpath}.anchor: unknown column {anchor!r}")
            anchor_col = parent_spec.columns[pcol]
        else:
            if anchor not in t.columns:
                raise MetadataError(f"{cpath}.anchor: unknown column {anchor!r}")
            anchor_col = t.columns[anchor]
            same_table_graph[cname] = anchor

        if anchor_col.type != "timestamp":
            raise MetadataError(f"{cpath}.anchor: anchored column {anchor!r} must be type timestamp")

        if t.columns[cname].type != "timestamp":
            raise MetadataError(f"{path}.columns.{cname}.type: temporal column must be type timestamp")

        _validate_distribution(f"{cpath}.delay", temp.delay, _DELAY_DIST_KINDS)

    # Cycle detection among same-table temporal anchors.
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str, chain: list[str]) -> None:
        if node not in same_table_graph:
            return
        if node in visiting:
            raise MetadataError(
                f"{path}.columns: temporal cycle detected: {' -> '.join(chain + [node])}"
            )
        if node in visited:
            return
        visiting.add(node)
        dfs(same_table_graph[node], chain + [node])
        visiting.discard(node)
        visited.add(node)

    for cname in same_table_graph:
        dfs(cname, [])


def _check_parent_cycles(md: Metadata) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(name: str, chain: list[str]) -> None:
        if name in visited:
            return
        if name in visiting:
            raise MetadataError(
                f"tables.{name}.parent: cycle detected in parent graph: "
                f"{' -> '.join(chain + [name])}"
            )
        visiting.add(name)
        parent = md.tables[name].parent
        if parent is not None:
            dfs(parent, chain + [name])
        visiting.discard(name)
        visited.add(name)

    for name in md.tables:
        dfs(name, [])


def validate(md: Metadata) -> None:
    if md.version != 1:
        raise MetadataError(f"version: must be 1 (got {md.version!r})")
    if not isinstance(md.seed, int) or isinstance(md.seed, bool):
        raise MetadataError(f"seed: must be an integer (got {md.seed!r})")
    if not md.tables:
        raise MetadataError("tables: must define at least one table")

    # Pass 1: role / rows / parent-existence / cardinality-presence / child_stride.
    for tname, t in md.tables.items():
        path = f"tables.{tname}"
        if t.source is not None:
            if not isinstance(t.source, str) or t.source == "" or "/" in t.source:
                raise MetadataError(
                    f"{path}.source: must be a non-empty string without '/' (got {t.source!r})"
                )
        if t.role not in ("root", "child"):
            raise MetadataError(f"{path}.role: must be 'root' or 'child' (got {t.role!r})")
        if t.role == "root":
            if t.parent is not None:
                raise MetadataError(f"{path}.parent: root table must not declare a parent")
            if t.cardinality is not None:
                raise MetadataError(f"{path}.cardinality: root table must not declare cardinality")
            if not isinstance(t.rows, int) or isinstance(t.rows, bool) or t.rows <= 0:
                raise MetadataError(f"{path}.rows: root table requires a positive integer 'rows'")
        else:  # child
            if t.parent is None or t.parent not in md.tables:
                raise MetadataError(f"{path}.parent: must name an existing table (got {t.parent!r})")
            if t.cardinality is None:
                raise MetadataError(f"{path}.cardinality: child table requires a cardinality spec")
            if not isinstance(t.child_stride, int) or isinstance(t.child_stride, bool) or t.child_stride <= 0:
                raise MetadataError(
                    f"{path}.child_stride: child table requires a positive integer 'child_stride'"
                )

    _check_parent_cycles(md)

    # Pass 2: columns, primary_key, cardinality contents, copulas, temporal, derived.
    for tname, t in md.tables.items():
        path = f"tables.{tname}"

        for cname, c in t.columns.items():
            cpath = f"{path}.columns.{cname}"
            if c.type not in _COLUMN_TYPES:
                raise MetadataError(f"{cpath}.type: must be one of {sorted(_COLUMN_TYPES)} (got {c.type!r})")

            sources = [s for s in (c.generator, c.distribution, c.temporal) if s is not None]
            if len(sources) != 1:
                raise MetadataError(
                    f"{cpath}: exactly one of generator/distribution/temporal must be set "
                    f"(got {len(sources)})"
                )

            if c.generator is not None:
                if c.generator in ("key", "parent_key"):
                    if c.generator == "parent_key" and t.role != "child":
                        raise MetadataError(f"{cpath}.generator: 'parent_key' is only valid on child tables")
                elif isinstance(c.generator, str) and c.generator.startswith("parent:"):
                    parent_col_name = c.generator[len("parent:"):]
                    if t.role != "child":
                        raise MetadataError(
                            f"{cpath}.generator: 'parent:{{column}}' is only valid on child tables"
                        )
                    if not parent_col_name:
                        raise MetadataError(f"{cpath}.generator: 'parent:' must name a column")
                    parent_table = md.tables.get(t.parent)
                    if parent_table is None:
                        raise MetadataError(f"{cpath}.generator: parent table {t.parent!r} not found")
                    parent_derived_names = {d.name for d in parent_table.derived}
                    if (
                        parent_col_name not in parent_table.columns
                        or parent_col_name in parent_derived_names
                    ):
                        raise MetadataError(
                            f"{cpath}.generator: parent column {parent_col_name!r} does not exist "
                            f"on table {t.parent!r} or is a derived column"
                        )
                    parent_col = parent_table.columns[parent_col_name]
                    if parent_col.type != c.type:
                        raise MetadataError(
                            f"{cpath}.generator: parent column {parent_col_name!r} has type "
                            f"{parent_col.type!r}, expected {c.type!r}"
                        )
                else:
                    raise MetadataError(
                        f"{cpath}.generator: must be 'key', 'parent_key', or 'parent:{{column}}' "
                        f"(got {c.generator!r})"
                    )

            if not isinstance(c.null_rate, (int, float)) or isinstance(c.null_rate, bool) or not (0.0 <= c.null_rate <= 1.0):
                raise MetadataError(f"{cpath}.null_rate: must be within [0, 1] (got {c.null_rate!r})")

            if c.clamp is not None:
                if len(c.clamp) != 2 or not (c.clamp[0] <= c.clamp[1]):
                    raise MetadataError(f"{cpath}.clamp: must be a 2-element [lo, hi] with lo <= hi")

            if c.distribution is not None:
                _validate_distribution(f"{cpath}.distribution", c.distribution)

            if c.reference is not None:
                if c.type != "int64":
                    raise MetadataError(
                        f"{cpath}.reference: column type must be 'int64' (got {c.type!r})"
                    )
                if c.distribution is None or c.distribution.kind not in _REFERENCE_DIST_KINDS:
                    raise MetadataError(
                        f"{cpath}.reference: column must have a distribution of kind "
                        f"{sorted(_REFERENCE_DIST_KINDS)} (a reference requires an integer "
                        "distribution; generator/temporal columns cannot carry a reference)"
                    )
                ref_table = md.tables.get(c.reference)
                if ref_table is None:
                    raise MetadataError(
                        f"{cpath}.reference: unknown table {c.reference!r}"
                    )
                if ref_table.role != "root":
                    raise MetadataError(
                        f"{cpath}.reference: referenced table {c.reference!r} must have "
                        f"role 'root' (got {ref_table.role!r})"
                    )

        if t.primary_key not in t.columns:
            raise MetadataError(f"{path}.primary_key: must name an existing column (got {t.primary_key!r})")
        pk_col = t.columns[t.primary_key]
        if pk_col.generator != "key":
            raise MetadataError(f"{path}.primary_key: column {t.primary_key!r} must have generator: key")

        if t.cardinality is not None:
            _validate_cardinality(f"{path}.cardinality", t.cardinality, t.child_stride)

        seen_copula_names: set[str] = set()
        for i, cop in enumerate(t.copulas):
            cpath = f"{path}.copulas.{cop.name}" if cop.name else f"{path}.copulas[{i}]"
            if not cop.name:
                raise MetadataError(f"{cpath}.name: required")
            if cop.name in seen_copula_names:
                raise MetadataError(f"{cpath}: duplicate copula name {cop.name!r}")
            seen_copula_names.add(cop.name)

            for col in cop.columns:
                if col not in t.columns:
                    raise MetadataError(f"{cpath}.columns: references unknown column {col!r}")
                if t.columns[col].distribution is None:
                    raise MetadataError(f"{cpath}.columns: column {col!r} must have a distribution")

            n = len(cop.columns)
            corr = cop.correlation
            if len(corr) != n or any(len(row) != n for row in corr):
                raise MetadataError(f"{cpath}.correlation: must be {n}x{n}")
            for i2 in range(n):
                if abs(corr[i2][i2] - 1.0) > 1e-9:
                    raise MetadataError(f"{cpath}.correlation: diagonal must be 1.0")
                for j2 in range(n):
                    if abs(corr[i2][j2] - corr[j2][i2]) > 1e-9:
                        raise MetadataError(f"{cpath}.correlation: must be symmetric")

        _validate_temporal(md, tname, t)

        for i, der in enumerate(t.derived):
            dpath = f"{path}.derived[{i}]"
            if not der.name:
                raise MetadataError(f"{dpath}.name: required")
            if not der.expr:
                raise MetadataError(f"{dpath}.expr: required")


# --------------------------------------------------------------------------
# Serialization (dataclasses -> plain yaml-safe dict) — exact inverse of
# parse_metadata / _parse_* above. See docs/ARCHITECTURE.md §2.
# --------------------------------------------------------------------------


def _distribution_spec_to_dict(dist: DistributionSpec) -> dict[str, Any]:
    d: dict[str, Any] = {"kind": dist.kind}
    d.update(dist.params)
    return d


def _cardinality_spec_to_dict(card: CardinalitySpec) -> dict[str, Any]:
    d: dict[str, Any] = {"kind": card.kind}
    d.update(card.params)
    return d


def _temporal_spec_to_dict(temp: TemporalSpec) -> dict[str, Any]:
    return {"anchor": temp.anchor, "delay": _distribution_spec_to_dict(temp.delay)}


def _copula_spec_to_dict(cop: CopulaSpec) -> dict[str, Any]:
    return {
        "name": cop.name,
        "columns": list(cop.columns),
        "correlation": [list(row) for row in cop.correlation],
    }


def _derived_spec_to_dict(der: DerivedSpec) -> dict[str, Any]:
    return {"name": der.name, "expr": der.expr}


def _column_spec_to_dict(col: ColumnSpec) -> dict[str, Any]:
    d: dict[str, Any] = {"type": col.type}
    if col.generator is not None:
        d["generator"] = col.generator
    if col.distribution is not None:
        d["distribution"] = _distribution_spec_to_dict(col.distribution)
    if col.temporal is not None:
        d["temporal"] = _temporal_spec_to_dict(col.temporal)
    if col.clamp is not None:
        d["clamp"] = [col.clamp[0], col.clamp[1]]
    if col.round:
        d["round"] = True
    if col.null_rate:
        d["null_rate"] = col.null_rate
    if col.reference is not None:
        d["reference"] = col.reference
    return d


def _table_spec_to_dict(t: TableSpec) -> dict[str, Any]:
    d: dict[str, Any] = {
        "role": t.role,
        "primary_key": t.primary_key,
        "columns": {cname: _column_spec_to_dict(c) for cname, c in t.columns.items()},
    }
    if t.source is not None:
        d["source"] = t.source
    if t.role == "root":
        d["rows"] = t.rows
    else:
        d["parent"] = t.parent
        d["cardinality"] = _cardinality_spec_to_dict(t.cardinality)
        d["child_stride"] = t.child_stride
    if t.copulas:
        d["copulas"] = [_copula_spec_to_dict(c) for c in t.copulas]
    if t.derived:
        d["derived"] = [_derived_spec_to_dict(d_) for d_ in t.derived]
    return d


def metadata_to_dict(md: Metadata) -> dict[str, Any]:
    """Serialize ``Metadata`` back to a plain, yaml-safe nested dict.

    Exact inverse of ``parse_metadata``: ``parse_metadata(metadata_to_dict(md))``
    must succeed and re-serialize to an equal dict.
    """
    return {
        "version": md.version,
        "seed": md.seed,
        "tables": {tname: _table_spec_to_dict(t) for tname, t in md.tables.items()},
    }
