"""Document rendering: shape parquet partitions into json / jsonl / xml files.

See docs/ARCHITECTURE.md §11 (normative). ``write_documents`` reads each
table's already-written Parquet partitions (via DuckDB, ``ORDER BY`` the
primary key so output is byte-identical regardless of partition count) and
renders one document file per table that declares a ``format`` in its
``TableSpec``. When ``format.schemas`` is set, the document is *shaped*
according to a minimal, deterministic JSON Schema / XSD walk (no external
schema-validation libraries): the schema is compiled once (from the first
row batch's column names, even when the table has zero rows, so compile-time
schema errors still surface) and then applied per row. ``validate_documents``
re-derives the same record counts and never raises for a missing/corrupt
document -- it reports a violation string instead.

Rows are never materialized in full: ``_iter_row_batches`` streams
``DEFAULT_FETCH_ROWS``-sized batches from DuckDB (``fetchmany`` over an
``ORDER BY <primary_key>`` cursor), and every renderer writes incrementally
from those batches instead of building a Python list of records or a full
XML ``ElementTree``. Memory stays O(batch) regardless of table size, so
multi-GB+ documents render without exhausting RAM. XML record counting in
``validate_documents`` similarly streams via ``xmlstream.count_xml_records``
instead of parsing a full DOM. For very large tables, ``kind: jsonl`` is the
recommended document kind: it streams one compact record per line with no
surrounding array, which is friendlier to downstream consumers than a single
multi-GB JSON array -- though the ``json`` array kind is still written
streamingly and remains byte-identical to a non-streaming ``json.dump``.

In-table **document columns** (``ColumnSpec.document``, docs/ARCHITECTURE.md
§11) are a different, complementary feature: a column whose per-row string
value is a JSON object or XML fragment serialized from that *same row's*
other columns, rather than a separately rendered per-table file. Generation
lives in ``compile_column_document``, called from ``Engine.generate_partition``:
it compiles the column's ``document`` spec (embedded sibling columns, optional
JSON Schema / XSD shaping) once into a per-row renderer function, reusing the
same schema-shaping machinery (``_compile_json_record_schema``,
``_compile_xsd_record``, ...) as table-level ``format``. The payload is
always a serialization of sibling columns already sampled elsewhere -- it is
never independently sampled -- so it stays in agreement with the relational
columns by construction, and rendering it costs no extra RNG draws beyond
the column's own ``null_rate`` mask.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import duckdb

from .backbone import ParquetBackbone
from .metadata import ColumnDocumentSpec, FormatSpec, Metadata, NestSpec, TableSpec
from .xmlstream import count_xml_records

_XS = "{http://www.w3.org/2001/XMLSchema}"

_EXTENSIONS = {"json": ".json", "jsonl": ".jsonl", "xml": ".xml"}

#: Rows fetched per DuckDB ``fetchmany`` call by ``_iter_row_batches``. A
#: module-level constant (rather than baked into renderer call sites) so
#: tests can tune batch size without touching every caller.
DEFAULT_FETCH_ROWS = 65536


class DocumentError(Exception):
    """Raised when document rendering fails (unresolvable schema, missing column...)."""


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def document_path(out_dir: str | Path, t: TableSpec) -> Path:
    """``{out}/{source}/{table}.{kind-extension}`` (no ``{source}`` level

    when ``t.source`` is None). ``t.format`` must be set.
    """
    if t.format is None:
        raise DocumentError(f"tables.{t.name}: document_path requires a format spec")
    ext = _EXTENSIONS[t.format.kind]
    parent = Path(out_dir) / t.source if t.source else Path(out_dir)
    return parent / f"{t.name}{ext}"


def _resolve_schema_path(base_dir: Path | None, s: str, tname: str) -> Path:
    p = Path(s)
    if not p.is_absolute():
        p = (base_dir or Path.cwd()) / p
    if not p.exists():
        raise DocumentError(f"tables.{tname}.format.schemas: schema file not found: {p}")
    return p


# --------------------------------------------------------------------------
# Row access (shared by all renderers)
# --------------------------------------------------------------------------


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _child_fk_column(child: TableSpec) -> str:
    return next(cn for cn, c in child.columns.items() if c.generator == "parent_key")


def _nest_alias(n: NestSpec) -> str:
    return n.alias or n.table


def _nest_columns(md: Metadata, n: NestSpec) -> list[str]:
    """Embedded child columns for a nest entry: explicit list, or every
    child column except its ``parent_key`` link (redundant inside the
    parent's record)."""
    child = md.tables[n.table]
    if n.columns:
        return list(n.columns)
    fk = _child_fk_column(child)
    return [cn for cn in child.columns if cn != fk]


def _nest_bindings(md: Metadata, nests: list[NestSpec]) -> dict[str, tuple[set[str], dict]]:
    """alias -> (child scalar column set, sub-bindings) for schema compilation."""
    return {
        _nest_alias(n): (set(_nest_columns(md, n)), _nest_bindings(md, n.nest))
        for n in nests
    }


def _nested_child_subquery(md: Metadata, backbone: ParquetBackbone, n: NestSpec) -> str:
    """Aggregate a nest entry's child rows into one ordered list of structs
    per referenced parent key (column ``__list``, keyed by ``__fk``)."""
    child = md.tables[n.table]
    fk = _child_fk_column(child)
    inner = _nested_select_sql(md, backbone, child, n.nest)
    fields = [f"{_q(c)} := c.{_q(c)}" for c in _nest_columns(md, n)]
    fields += [f"{_q(_nest_alias(sub))} := c.{_q(_nest_alias(sub))}" for sub in n.nest]
    return (
        f"SELECT {_q(fk)} AS __fk, "
        f"list(struct_pack({', '.join(fields)}) ORDER BY c.{_q(child.primary_key)}) AS __list "
        f"FROM ({inner}) c GROUP BY {_q(fk)}"
    )


def _nested_select_sql(
    md: Metadata, backbone: ParquetBackbone, t: TableSpec, nests: list[NestSpec]
) -> str:
    """SELECT over ``t``'s partitions with one list-of-structs column per
    nest entry appended (empty list for childless parents), ordered by
    primary key — DuckDB does all the relational grouping out-of-core."""
    glob = backbone.table_glob(t.name, t.source)
    base = f"SELECT * FROM read_parquet('{glob}')"
    if not nests:
        return f"{base} ORDER BY {_q(t.primary_key)}"
    select_cols = ["p.*"]
    joins = []
    for i, n in enumerate(nests):
        sub = _nested_child_subquery(md, backbone, n)
        joins.append(f"LEFT JOIN ({sub}) n{i} ON p.{_q(t.primary_key)} = n{i}.__fk")
        select_cols.append(f"coalesce(n{i}.__list, []) AS {_q(_nest_alias(n))}")
    return (
        f"SELECT {', '.join(select_cols)} FROM ({base}) p "
        + " ".join(joins)
        + f" ORDER BY p.{_q(t.primary_key)}"
    )


def _iter_row_batches(
    sql: str, batch_rows: int = DEFAULT_FETCH_ROWS
) -> Iterator[tuple[list[str], list[tuple]]]:
    """Stream ``(col_names, rows_batch)`` tuples for ``sql`` with O(batch)
    memory (nested list-of-struct columns arrive as Python lists of dicts).

    The DuckDB connection stays open across iteration and is closed in
    ``finally`` whether the generator runs to completion or is abandoned
    early (garbage-collected / ``.close()``d by the caller).

    Column names are always available from the first yielded tuple, even
    when the table has zero rows: on an empty result, exactly one
    ``(col_names, [])`` batch is yielded so callers can still compile a
    schema plan (and surface compile-time errors) against an empty table.
    """
    con = duckdb.connect()
    try:
        result = con.execute(sql)
        col_names = [d[0] for d in result.description]
        yielded_any = False
        while True:
            batch = result.fetchmany(batch_rows)
            if not batch:
                if not yielded_any:
                    yield col_names, []
                break
            yielded_any = True
            yield col_names, batch
    finally:
        con.close()


def _xml_text(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _singular(name: str) -> str:
    return name[:-1] if name.endswith("s") else name


# --------------------------------------------------------------------------
# json / jsonl, no schema: fully delegated to DuckDB
# --------------------------------------------------------------------------


def _render_json_plain(
    metadata: Metadata, backbone: ParquetBackbone, t: TableSpec, path: Path
) -> None:
    # Nested child collections (format.nest) ride along natively: DuckDB
    # serializes the aggregated list-of-struct columns as JSON arrays.
    sql = _nested_select_sql(metadata, backbone, t, t.format.nest)
    array_opt = ", ARRAY true" if t.format.kind == "json" else ""
    con = duckdb.connect()
    try:
        con.execute(f"COPY ({sql}) TO '{path}' (FORMAT JSON{array_opt})")
    finally:
        con.close()


# --------------------------------------------------------------------------
# json / jsonl, with schema(s): JSON Schema shaping
# --------------------------------------------------------------------------


@dataclass
class _LeafPlan:
    column: str
    required: bool
    type: str | None


@dataclass
class _ObjectPlan:
    required: bool
    children: list[tuple[str, "_LeafPlan | _ObjectPlan"]] = field(default_factory=list)


def _leaf_type(node: Any) -> str | None:
    if not isinstance(node, dict):
        return None
    t = node.get("type")
    if isinstance(t, list):
        for x in t:
            if x != "null":
                return x
        return None
    return t


def _walk_json_pointer(doc: Any, pointer: str, tname: str, ref: str) -> Any:
    if pointer in ("", "/"):
        return doc
    parts = pointer.split("/")
    if parts and parts[0] == "":
        parts = parts[1:]
    node = doc
    for raw in parts:
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            try:
                node = node[int(key)]
            except (ValueError, IndexError) as e:
                raise DocumentError(
                    f"tables.{tname}.format: unresolvable $ref {ref!r}"
                ) from e
        elif isinstance(node, dict):
            if key not in node:
                raise DocumentError(f"tables.{tname}.format: unresolvable $ref {ref!r}")
            node = node[key]
        else:
            raise DocumentError(f"tables.{tname}.format: unresolvable $ref {ref!r}")
    return node


def _deref_json(
    node: Any, doc: Any, store: dict[str, Any], tname: str, prop_path: str
) -> tuple[Any, Any]:
    seen: set[str] = set()
    while isinstance(node, dict) and "$ref" in node:
        ref = node["$ref"]
        if ref in seen:
            raise DocumentError(
                f"tables.{tname}.format: circular $ref {ref!r} while resolving {prop_path!r}"
            )
        seen.add(ref)
        if "#" in ref:
            a, p = ref.split("#", 1)
        else:
            a, p = ref, ""
        if a == "":
            target_doc = doc
        else:
            target_doc = store.get(a)
            if target_doc is None:
                target_doc = store.get(Path(a).name)
            if target_doc is None:
                raise DocumentError(
                    f"tables.{tname}.format: unresolvable $ref {ref!r} (schema {a!r} not loaded)"
                )
        node = _walk_json_pointer(target_doc, p, tname, ref)
        doc = target_doc
    return node, doc


@dataclass
class _ArrayPlan:
    """A schema ``array`` property bound to a nested child collection
    (format.nest): the row carries a list of child-row dicts under
    ``column``, each shaped by ``item``."""

    column: str
    required: bool
    item: "_LeafPlan | _ObjectPlan | _ArrayPlan"


def _compile_json_node(
    node: Any,
    doc: Any,
    store: dict[str, Any],
    tname: str,
    prop_path: str,
    required: bool,
    columns: set[str],
    nests: dict[str, tuple[set[str], dict]] | None = None,
) -> "_LeafPlan | _ObjectPlan | _ArrayPlan | None":
    node, doc = _deref_json(node, doc, store, tname, prop_path)
    nests = nests or {}

    pname = prop_path.rsplit(".", 1)[-1] if prop_path else prop_path
    is_array = isinstance(node, dict) and (
        node.get("type") == "array" or "items" in node
    )
    if is_array and prop_path:
        if pname not in nests:
            if required:
                raise DocumentError(
                    f"tables.{tname}.format: schema array property {prop_path!r} "
                    "has no matching nested child (format.nest)"
                )
            return None
        item_cols, sub_nests = nests[pname]
        item_plan = _compile_json_node(
            node.get("items", {}),
            doc,
            store,
            tname,
            f"{prop_path}[]",
            True,
            item_cols,
            sub_nests,
        )
        return _ArrayPlan(column=pname, required=required, item=item_plan)

    is_object = isinstance(node, dict) and (
        "properties" in node or node.get("type") == "object"
    )
    if is_object:
        props = node.get("properties") or {}
        req_names = set(node.get("required") or [])
        children: list[tuple[str, _LeafPlan | _ObjectPlan | _ArrayPlan]] = []
        for cprop, pnode in props.items():
            child_path = f"{prop_path}.{cprop}" if prop_path else cprop
            child_required = cprop in req_names
            plan = _compile_json_node(
                pnode, doc, store, tname, child_path, child_required, columns, nests
            )
            if plan is not None:
                children.append((cprop, plan))
        return _ObjectPlan(required=required, children=children)

    if pname not in columns:
        if required:
            raise DocumentError(
                f"tables.{tname}.format: schema property {prop_path!r} has no matching column"
            )
        return None
    return _LeafPlan(column=pname, required=required, type=_leaf_type(node))


def _compile_json_record_schema(
    schemas: list[str],
    base_dir: Path | None,
    tname: str,
    columns: set[str],
    nests: dict[str, tuple[set[str], dict]] | None = None,
) -> "_LeafPlan | _ObjectPlan":
    store: dict[str, Any] = {}
    primary_doc: Any = None
    for i, s in enumerate(schemas):
        spath = _resolve_schema_path(base_dir, s, tname)
        with open(spath) as f:
            doc = json.load(f)
        store[spath.name] = doc
        if isinstance(doc, dict) and doc.get("$id"):
            store[doc["$id"]] = doc
        if i == 0:
            primary_doc = doc

    if isinstance(primary_doc, dict) and primary_doc.get("type") == "array":
        record_node = primary_doc.get("items", {})
    else:
        record_node = primary_doc

    return _compile_json_node(
        record_node, primary_doc, store, tname, "", True, columns, nests
    )


def _coerce_json_scalar(value: Any, schema_type: str | None) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if schema_type == "integer":
        return int(value)
    if schema_type == "number":
        return float(value)
    if schema_type == "boolean":
        return bool(value)
    if schema_type == "string":
        return str(value)
    return value


def _shape_json_node(
    plan: "_LeafPlan | _ObjectPlan | _ArrayPlan", row: dict[str, Any]
) -> tuple[bool, Any]:
    if isinstance(plan, _ArrayPlan):
        items = row.get(plan.column) or []
        return True, [_shape_json_node(plan.item, item)[1] for item in items]
    if isinstance(plan, _LeafPlan):
        value = row.get(plan.column)
        if value is None:
            if plan.required:
                return True, None
            return False, None
        return True, _coerce_json_scalar(value, plan.type)

    d: dict[str, Any] = {}
    for pname, child in plan.children:
        included, value = _shape_json_node(child, row)
        if included:
            d[pname] = value
    if not d and not plan.required:
        return False, None
    return True, d


def _render_json_shaped(
    metadata: Metadata, backbone: ParquetBackbone, t: TableSpec, path: Path
) -> None:
    sql = _nested_select_sql(metadata, backbone, t, t.format.nest)
    batches = _iter_row_batches(sql, batch_rows=DEFAULT_FETCH_ROWS)
    col_names, first_batch = next(batches)
    # Compile (and raise on unresolvable schema properties) before opening
    # the output file, matching the old fetch-then-compile-then-write order.
    plan = _compile_json_record_schema(
        t.format.schemas,
        metadata.base_dir,
        t.name,
        set(col_names),
        _nest_bindings(metadata, t.format.nest),
    )

    def _all_batches() -> Iterator[list[tuple]]:
        yield first_batch
        for _cols, batch in batches:
            yield batch

    if t.format.kind == "json":
        with open(path, "w") as f:
            wrote_any = False
            for batch in _all_batches():
                for row in batch:
                    row_dict = dict(zip(col_names, row))
                    _, record = _shape_json_node(plan, row_dict)
                    rendered = "\n".join(
                        "  " + line for line in json.dumps(record, indent=2).splitlines()
                    )
                    f.write("[\n" if not wrote_any else ",\n")
                    f.write(rendered)
                    wrote_any = True
            f.write("\n]\n" if wrote_any else "[]\n")
    else:
        with open(path, "w") as f:
            for batch in _all_batches():
                for row in batch:
                    row_dict = dict(zip(col_names, row))
                    _, record = _shape_json_node(plan, row_dict)
                    f.write(json.dumps(record, separators=(",", ":")))
                    f.write("\n")


# --------------------------------------------------------------------------
# xml, no schema
# --------------------------------------------------------------------------


def _append_plain_nested(parent_el: ET.Element, metadata: Metadata, n: NestSpec, items: list) -> None:
    """Flat-XML convention for a nested child collection: a ``<{alias}>``
    container holding one ``<{singular(alias)}>`` element per child row."""
    alias = _nest_alias(n)
    container = ET.SubElement(parent_el, alias)
    item_name = _singular(alias)
    sub_aliases = {_nest_alias(s): s for s in n.nest}
    for item in items or []:
        item_el = ET.SubElement(container, item_name)
        for key, value in item.items():
            if key in sub_aliases:
                _append_plain_nested(item_el, metadata, sub_aliases[key], value)
            elif value is not None:
                child = ET.SubElement(item_el, key)
                child.text = _xml_text(value)


def _render_xml_plain(
    metadata: Metadata, backbone: ParquetBackbone, t: TableSpec, path: Path
) -> None:
    fmt = t.format
    root_name = fmt.root or t.name
    record_name = fmt.record or _singular(t.name)
    sql = _nested_select_sql(metadata, backbone, t, fmt.nest)
    aliases = {_nest_alias(n): n for n in fmt.nest}

    def _records() -> Iterator[ET.Element]:
        for col_names, batch in _iter_row_batches(sql, batch_rows=DEFAULT_FETCH_ROWS):
            for row in batch:
                rec_el = ET.Element(record_name)
                for cname, value in zip(col_names, row):
                    if cname in aliases:
                        _append_plain_nested(rec_el, metadata, aliases[cname], value)
                    elif value is not None:
                        child = ET.SubElement(rec_el, cname)
                        child.text = _xml_text(value)
                yield rec_el

    _write_xml_stream(root_name, _records(), path)


def _write_xml_stream(root_name: str, record_elements: Iterator[ET.Element], path: Path) -> None:
    """Stream ``<?xml ...?><root>...records...</root>`` byte-identically to
    the old build-full-tree-then-``ET.indent`` writer, without ever holding
    more than one record ``Element`` in memory.

    Each record subtree is indented independently with ``ET.indent`` (the
    2-space nesting is the same whether the element is a lone tree or a
    subtree of a bigger one), serialized, then re-indented by one more level
    (the record's position under ``<root>``) and written immediately.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("<?xml version='1.0' encoding='utf-8'?>\n")
        wrote_any = False
        for rec_el in record_elements:
            if not wrote_any:
                f.write(f"<{root_name}>")
                wrote_any = True
            ET.indent(rec_el, space="  ")
            rec_xml = ET.tostring(rec_el, encoding="unicode")
            f.write("\n")
            f.write("\n".join("  " + line for line in rec_xml.splitlines()))
        if wrote_any:
            f.write(f"\n</{root_name}>\n")
        else:
            f.write(f"<{root_name} />\n")


# --------------------------------------------------------------------------
# xml, with schema(s): XSD shaping
# --------------------------------------------------------------------------


@dataclass
class _XLeafPlan:
    name: str
    column: str
    required: bool


@dataclass
class _XObjectPlan:
    name: str
    required: bool
    children: list["_XLeafPlan | _XObjectPlan"] = field(default_factory=list)


def _local_name(qname: str) -> str:
    return qname.split(":")[-1]


def _load_xsd_documents(
    fmt: FormatSpec, base_dir: Path | None, tname: str
) -> list[ET.Element]:
    loaded: dict[Path, ET.Element] = {}
    roots: list[ET.Element] = []

    def _load(path: Path) -> None:
        rp = path.resolve()
        if rp in loaded:
            return
        if not path.exists():
            raise DocumentError(f"tables.{tname}.format.schemas: schema file not found: {path}")
        root = ET.parse(path).getroot()
        loaded[rp] = root
        roots.append(root)
        for tag in ("include", "import"):
            for el in root.findall(f"{_XS}{tag}"):
                loc = el.get("schemaLocation")
                if not loc:
                    continue
                loc_path = Path(loc)
                if not loc_path.is_absolute():
                    loc_path = path.parent / loc_path
                _load(loc_path)

    for s in fmt.schemas:
        p = _resolve_schema_path(base_dir, s, tname)
        _load(p)

    return roots


def _collect_xsd_globals(
    roots: list[ET.Element],
) -> tuple[dict[str, ET.Element], dict[str, ET.Element]]:
    elements: dict[str, ET.Element] = {}
    complex_types: dict[str, ET.Element] = {}
    for root in roots:
        for el in root.findall(f"{_XS}element"):
            name = el.get("name")
            if name:
                elements[name] = el
        for ct in root.findall(f"{_XS}complexType"):
            name = ct.get("name")
            if name:
                complex_types[name] = ct
    return elements, complex_types


def _xsd_complex_type_of(
    el: ET.Element, complex_types: dict[str, ET.Element]
) -> ET.Element | None:
    inline = el.find(f"{_XS}complexType")
    if inline is not None:
        return inline
    type_attr = el.get("type")
    if type_attr:
        return complex_types.get(_local_name(type_attr))
    return None


def _find_unbounded_particle(
    wrapper_el: ET.Element,
    elements: dict[str, ET.Element],
    complex_types: dict[str, ET.Element],
    tname: str,
) -> ET.Element | None:
    ct = _xsd_complex_type_of(wrapper_el, complex_types)
    if ct is None:
        return None
    seq = ct.find(f"{_XS}sequence")
    if seq is None:
        return None
    particles = seq.findall(f"{_XS}element")
    unbounded = [p for p in particles if p.get("maxOccurs") == "unbounded"]
    if len(particles) != 1 or len(unbounded) != 1:
        return None
    particle = unbounded[0]
    ref = particle.get("ref")
    if ref:
        target = elements.get(_local_name(ref))
        if target is None:
            raise DocumentError(
                f"tables.{tname}.format: element ref {ref!r} not found among global elements"
            )
        return target
    return particle


def _find_record_element(
    fmt: FormatSpec,
    t: TableSpec,
    elements: dict[str, ET.Element],
    complex_types: dict[str, ET.Element],
) -> tuple[ET.Element, str | None]:
    """Returns (record element declaration, wrapper name used by rule 2 or None)."""
    if fmt.record:
        el = elements.get(fmt.record)
        if el is None:
            raise DocumentError(
                f"tables.{t.name}.format.record: no global element named {fmt.record!r}"
            )
        return el, None

    candidate = fmt.root or t.name
    wrapper_el = elements.get(candidate)
    if wrapper_el is not None:
        unbounded = _find_unbounded_particle(wrapper_el, elements, complex_types, t.name)
        if unbounded is not None:
            return unbounded, candidate

    if len(elements) == 1:
        (only_el,) = elements.values()
        return only_el, None

    raise DocumentError(
        f"tables.{t.name}.format: cannot determine the record element automatically "
        "(no unique global element / unbounded wrapper particle found); set format.record"
    )


@dataclass
class _XArrayPlan:
    """A repeated (or wrapped-repeated) XSD element bound to a nested child
    collection (format.nest): the row carries a list of child-row dicts
    under ``column``. With ``container`` set, the repeated ``item`` elements
    sit inside a ``<container>`` element; otherwise they repeat in place."""

    column: str
    container: str | None
    item: "_XObjectPlan"


def _compile_xsd_sequence(
    ct: ET.Element,
    elements: dict[str, ET.Element],
    complex_types: dict[str, ET.Element],
    tname: str,
    columns: set[str],
    nests: dict[str, tuple[set[str], dict]] | None = None,
) -> list["_XLeafPlan | _XObjectPlan | _XArrayPlan"]:
    seq = ct.find(f"{_XS}sequence")
    if seq is None:
        seq = ct.find(f"{_XS}all")
    if seq is None:
        return []
    children: list[_XLeafPlan | _XObjectPlan | _XArrayPlan] = []
    for particle_el in seq.findall(f"{_XS}element"):
        plan = _compile_xsd_particle(
            particle_el, elements, complex_types, tname, columns, nests
        )
        if plan is not None:
            children.append(plan)
    return children


def _compile_xsd_item_plan(
    name: str,
    item_el: ET.Element,
    elements: dict[str, ET.Element],
    complex_types: dict[str, ET.Element],
    tname: str,
    binding: tuple[set[str], dict],
) -> _XObjectPlan:
    item_cols, sub_nests = binding
    ct = _xsd_complex_type_of(item_el, complex_types)
    if ct is None:
        raise DocumentError(
            f"tables.{tname}.format: nested element {name!r} has no complex content to shape"
        )
    children = _compile_xsd_sequence(ct, elements, complex_types, tname, item_cols, sub_nests)
    return _XObjectPlan(name=item_el.get("name") or name, required=True, children=children)


def _compile_xsd_particle(
    particle_el: ET.Element,
    elements: dict[str, ET.Element],
    complex_types: dict[str, ET.Element],
    tname: str,
    columns: set[str],
    nests: dict[str, tuple[set[str], dict]] | None = None,
) -> "_XLeafPlan | _XObjectPlan | _XArrayPlan | None":
    nests = nests or {}
    ref = particle_el.get("ref")
    if ref:
        target = elements.get(_local_name(ref))
        if target is None:
            raise DocumentError(
                f"tables.{tname}.format: element ref {ref!r} not found among global elements"
            )
        name = target.get("name")
        type_source = target
        min_occurs = particle_el.get("minOccurs", target.get("minOccurs", "1"))
    else:
        name = particle_el.get("name")
        type_source = particle_el
        min_occurs = particle_el.get("minOccurs", "1")

    required = min_occurs != "0"
    unbounded = particle_el.get("maxOccurs") == "unbounded"

    if name in nests:
        # Bound to a nested child collection: either the element itself
        # repeats (maxOccurs="unbounded"), or it is a container wrapping a
        # single unbounded item element.
        if unbounded:
            return _XArrayPlan(
                column=name,
                container=None,
                item=_compile_xsd_item_plan(
                    name, type_source, elements, complex_types, tname, nests[name]
                ),
            )
        inner = _find_unbounded_particle(type_source, elements, complex_types, tname)
        if inner is not None:
            return _XArrayPlan(
                column=name,
                container=name,
                item=_compile_xsd_item_plan(
                    inner.get("name"), inner, elements, complex_types, tname, nests[name]
                ),
            )
        raise DocumentError(
            f"tables.{tname}.format: element {name!r} matches a nested child but is "
            "neither maxOccurs=\"unbounded\" nor a wrapper of a single unbounded element"
        )

    ct = _xsd_complex_type_of(type_source, complex_types)

    if ct is None:
        if name not in columns:
            if required:
                raise DocumentError(
                    f"tables.{tname}.format: schema element {name!r} has no matching column"
                )
            return None
        return _XLeafPlan(name=name, column=name, required=required)

    children = _compile_xsd_sequence(ct, elements, complex_types, tname, columns, nests)
    return _XObjectPlan(name=name, required=required, children=children)


def _compile_xsd_record(
    record_el: ET.Element,
    elements: dict[str, ET.Element],
    complex_types: dict[str, ET.Element],
    tname: str,
    columns: set[str],
    nests: dict[str, tuple[set[str], dict]] | None = None,
) -> _XObjectPlan:
    name = record_el.get("name")
    ct = _xsd_complex_type_of(record_el, complex_types)
    if ct is None:
        raise DocumentError(
            f"tables.{tname}.format: record element {name!r} has no complex content to shape"
        )
    children = _compile_xsd_sequence(ct, elements, complex_types, tname, columns, nests)
    return _XObjectPlan(name=name, required=True, children=children)


def _render_xsd_node(
    plan: "_XLeafPlan | _XObjectPlan | _XArrayPlan", row: dict[str, Any]
) -> ET.Element | list[ET.Element] | None:
    if isinstance(plan, _XArrayPlan):
        items = row.get(plan.column) or []
        rendered = [_render_xsd_node(plan.item, item) for item in items]
        rendered = [el for el in rendered if el is not None]
        if plan.container is not None:
            container = ET.Element(plan.container)
            container.extend(rendered)
            return container
        return rendered

    if isinstance(plan, _XLeafPlan):
        value = row.get(plan.column)
        if value is None:
            return None
        el = ET.Element(plan.name)
        el.text = _xml_text(value)
        return el

    el = ET.Element(plan.name)
    any_child = False
    for child in plan.children:
        child_el = _render_xsd_node(child, row)
        if child_el is None:
            continue
        if isinstance(child_el, list):
            el.extend(child_el)
            any_child = any_child or bool(child_el)
        else:
            el.append(child_el)
            any_child = True
    if not any_child and not plan.required:
        return None
    return el


def _render_xml_shaped(
    metadata: Metadata, backbone: ParquetBackbone, t: TableSpec, path: Path
) -> None:
    fmt = t.format
    roots = _load_xsd_documents(fmt, metadata.base_dir, t.name)
    elements, complex_types = _collect_xsd_globals(roots)
    record_el, wrapper_name = _find_record_element(fmt, t, elements, complex_types)
    root_name = fmt.root or wrapper_name or t.name

    sql = _nested_select_sql(metadata, backbone, t, fmt.nest)
    batches = _iter_row_batches(sql, batch_rows=DEFAULT_FETCH_ROWS)
    col_names, first_batch = next(batches)
    # Compile (and raise on unresolvable schema elements) before opening the
    # output file, matching the old fetch-then-compile-then-write order.
    plan = _compile_xsd_record(
        record_el,
        elements,
        complex_types,
        t.name,
        set(col_names),
        _nest_bindings(metadata, fmt.nest),
    )

    def _all_batches() -> Iterator[list[tuple]]:
        yield first_batch
        for _cols, batch in batches:
            yield batch

    def _records() -> Iterator[ET.Element]:
        for batch in _all_batches():
            for row in batch:
                row_dict = dict(zip(col_names, row))
                rec_el = _render_xsd_node(plan, row_dict)
                if rec_el is not None:
                    yield rec_el

    _write_xml_stream(root_name, _records(), path)


# --------------------------------------------------------------------------
# In-table document columns (docs/ARCHITECTURE.md §11): a column whose per-row
# value is a JSON object / XML fragment rendered from that row's own sibling
# columns. See the module docstring for the design rationale.
# --------------------------------------------------------------------------


def _resolve_document_columns(t: TableSpec, cname: str) -> list[str]:
    """The embedded sibling column names for document column ``cname``, in
    the order they should appear in the rendered payload.

    Explicit ``document.columns`` is honored verbatim; the empty-list DEFAULT
    is every sibling column with no ``document`` spec of its own, in table
    declaration order, excluding ``cname`` itself.
    """
    spec = t.columns[cname].document
    if spec.columns:
        return list(spec.columns)
    return [
        name
        for name, c in t.columns.items()
        if name != cname and c.document is None
    ]


def _resolve_document_record_element(
    spec: ColumnDocumentSpec,
    t: TableSpec,
    elements: dict[str, ET.Element],
    complex_types: dict[str, ET.Element],
) -> ET.Element:
    """Resolve the XSD global element that shapes a document column's
    payload fragment.

    If ``spec.root`` names a global element directly, that element *is* the
    record (unlike table-level ``format.root``, a document column's XML
    fragment is never a repeated-wrapper list, so no "unbounded particle"
    unwrapping applies here). Otherwise fall back to the same
    wrapper/unique-global-element discovery ``_find_record_element`` uses for
    table-level documents, with candidate name ``spec.root or
    _singular(t.name)`` -- expressed as a small shim ``FormatSpec`` so the
    discovery logic itself is not duplicated.
    """
    if spec.root is not None:
        el = elements.get(spec.root)
        if el is not None:
            return el
    candidate = spec.root or _singular(t.name)
    shim_fmt = FormatSpec(kind="xml", root=candidate)
    record_el, _wrapper_name = _find_record_element(shim_fmt, t, elements, complex_types)
    return record_el


def compile_column_document(
    metadata: Metadata, t: TableSpec, cname: str
) -> Callable[[dict[str, Any]], str]:
    """Compile the ``document`` spec of column ``cname`` into a per-row
    renderer: ``row`` (embedded column name -> python value: int/float/bool/
    str/datetime/None) -> document string.

    Schema compilation (when ``document.schemas`` is set) happens eagerly,
    here, so an unresolvable schema surfaces as a ``DocumentError`` at compile
    time rather than on the first rendered row.
    """
    spec = t.columns[cname].document
    embedded = _resolve_document_columns(t, cname)

    if spec.kind == "json":
        if spec.schemas:
            plan = _compile_json_record_schema(
                spec.schemas, metadata.base_dir, t.name, set(embedded)
            )

            def _render_json(row: dict[str, Any]) -> str:
                _, record = _shape_json_node(plan, row)
                return json.dumps(record, separators=(",", ":"))

            return _render_json

        def _render_json_plain_row(row: dict[str, Any]) -> str:
            record = {c: _coerce_json_scalar(row.get(c), None) for c in embedded}
            return json.dumps(record, separators=(",", ":"))

        return _render_json_plain_row

    # kind == "xml"
    if spec.schemas:
        shim_fmt = FormatSpec(kind="xml", schemas=spec.schemas)
        roots = _load_xsd_documents(shim_fmt, metadata.base_dir, t.name)
        elements, complex_types = _collect_xsd_globals(roots)
        record_el = _resolve_document_record_element(spec, t, elements, complex_types)
        plan = _compile_xsd_record(record_el, elements, complex_types, t.name, set(embedded))

        def _render_xml_shaped_row(row: dict[str, Any]) -> str:
            el = _render_xsd_node(plan, row)
            return ET.tostring(el, encoding="unicode")

        return _render_xml_shaped_row

    root_name = spec.root or _singular(t.name)

    def _render_xml_plain_row(row: dict[str, Any]) -> str:
        el = ET.Element(root_name)
        for c in embedded:
            value = row.get(c)
            if value is None:
                continue
            child = ET.SubElement(el, c)
            child.text = _xml_text(value)
        return ET.tostring(el, encoding="unicode")

    return _render_xml_plain_row


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def _render_document(
    metadata: Metadata, backbone: ParquetBackbone, t: TableSpec, path: Path
) -> None:
    fmt = t.format
    if fmt.kind in ("json", "jsonl"):
        if fmt.schemas:
            _render_json_shaped(metadata, backbone, t, path)
        else:
            _render_json_plain(metadata, backbone, t, path)
    else:  # xml
        if fmt.schemas:
            _render_xml_shaped(metadata, backbone, t, path)
        else:
            _render_xml_plain(metadata, backbone, t, path)


def write_documents(metadata: Metadata, out_dir: str | Path) -> dict[str, Path]:
    """Render one document file per table with a ``format`` spec.

    Rows are read from the already-written Parquet partitions (see
    ``ParquetBackbone``); tables without ``format`` are skipped. Returns
    ``{table_name: document_path}``.
    """
    out_dir = Path(out_dir)
    backbone = ParquetBackbone(out_dir)
    results: dict[str, Path] = {}
    for tname in metadata.table_order():
        t = metadata.tables[tname]
        if t.format is None:
            continue
        path = document_path(out_dir, t)
        path.parent.mkdir(parents=True, exist_ok=True)
        _render_document(metadata, backbone, t, path)
        results[tname] = path
    return results


def validate_documents(metadata: Metadata, out_dir: str | Path) -> list[str]:
    """Human-readable violation strings (empty list = OK).

    Compares each table's document record count against its Parquet
    partitions' row count. Never raises: a missing or corrupt document file
    becomes a violation string instead.
    """
    out_dir = Path(out_dir)
    backbone = ParquetBackbone(out_dir)
    violations: list[str] = []
    con = duckdb.connect()
    try:
        for tname in metadata.table_order():
            t = metadata.tables[tname]
            if t.format is None:
                continue
            path = document_path(out_dir, t)
            if not path.exists():
                violations.append(f"{tname}: document {path} is missing")
                continue

            glob = backbone.table_glob(tname, t.source)
            try:
                (expected,) = con.execute(
                    f"SELECT count(*) FROM read_parquet('{glob}')"
                ).fetchone()
            except Exception as e:  # pragma: no cover - defensive
                violations.append(
                    f"{tname}: could not read parquet partitions to validate {path}: {e}"
                )
                continue

            if t.format.kind in ("json", "jsonl"):
                try:
                    (n,) = con.execute(
                        f"SELECT count(*) FROM read_json_auto('{path}')"
                    ).fetchone()
                except Exception as e:
                    violations.append(
                        f"{tname}: document {path} could not be read as JSON: {e}"
                    )
                    continue
            else:  # xml
                try:
                    n = count_xml_records(path)
                except (ET.ParseError, ValueError) as e:
                    violations.append(
                        f"{tname}: document {path} could not be parsed as XML: {e}"
                    )
                    continue

            if n != expected:
                violations.append(
                    f"{tname}: document {path} has {n} records, expected {expected}"
                )

            # Nested child collections: the summed nested lengths must equal
            # the child table's Parquet row count (every child row appears
            # exactly once under its parent). JSON kinds only — XML nested
            # counting would need schema-aware tag walking.
            if t.format.kind in ("json", "jsonl"):
                for nest in t.format.nest:
                    child = metadata.tables[nest.table]
                    child_glob = backbone.table_glob(nest.table, child.source)
                    alias = nest.alias or nest.table
                    try:
                        (child_expected,) = con.execute(
                            f"SELECT count(*) FROM read_parquet('{child_glob}')"
                        ).fetchone()
                        (nested_n,) = con.execute(
                            f"SELECT coalesce(sum(len({_q(alias)})), 0) "
                            f"FROM read_json_auto('{path}')"
                        ).fetchone()
                    except Exception as e:
                        violations.append(
                            f"{tname}: could not count nested {alias!r} records "
                            f"in {path}: {e}"
                        )
                        continue
                    if nested_n != child_expected:
                        violations.append(
                            f"{tname}: document {path} nests {nested_n} {alias!r} "
                            f"records, expected {child_expected} ({nest.table})"
                        )
    finally:
        con.close()

    return violations
