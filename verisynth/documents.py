"""Document rendering: shape parquet partitions into json / jsonl / xml files.

See docs/ARCHITECTURE.md §11 (normative). ``write_documents`` reads each
table's already-written Parquet partitions (via DuckDB, ``ORDER BY`` the
primary key so output is byte-identical regardless of partition count) and
renders one document file per table that declares a ``format`` in its
``TableSpec``. When ``format.schemas`` is set, the document is *shaped*
according to a minimal, deterministic JSON Schema / XSD walk (no external
schema-validation libraries): the schema is compiled once into a small plan
and then applied per row. ``validate_documents`` re-derives the same record
counts and never raises for a missing/corrupt document -- it reports a
violation string instead.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from .backbone import ParquetBackbone
from .metadata import FormatSpec, Metadata, TableSpec

_XS = "{http://www.w3.org/2001/XMLSchema}"

_EXTENSIONS = {"json": ".json", "jsonl": ".jsonl", "xml": ".xml"}


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


def _ordered_rows(glob: str, pk: str) -> tuple[list[str], list[tuple]]:
    con = duckdb.connect()
    try:
        result = con.execute(f"SELECT * FROM read_parquet('{glob}') ORDER BY {pk}")
        col_names = [d[0] for d in result.description]
        rows = result.fetchall()
    finally:
        con.close()
    return col_names, rows


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


def _render_json_plain(t: TableSpec, glob: str, path: Path) -> None:
    array_opt = ", ARRAY true" if t.format.kind == "json" else ""
    con = duckdb.connect()
    try:
        con.execute(
            f"COPY (SELECT * FROM read_parquet('{glob}') ORDER BY {t.primary_key}) "
            f"TO '{path}' (FORMAT JSON{array_opt})"
        )
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


def _compile_json_node(
    node: Any,
    doc: Any,
    store: dict[str, Any],
    tname: str,
    prop_path: str,
    required: bool,
    columns: set[str],
) -> "_LeafPlan | _ObjectPlan | None":
    node, doc = _deref_json(node, doc, store, tname, prop_path)
    is_object = isinstance(node, dict) and (
        "properties" in node or node.get("type") == "object"
    )
    if is_object:
        props = node.get("properties") or {}
        req_names = set(node.get("required") or [])
        children: list[tuple[str, _LeafPlan | _ObjectPlan]] = []
        for pname, pnode in props.items():
            child_path = f"{prop_path}.{pname}" if prop_path else pname
            child_required = pname in req_names
            plan = _compile_json_node(
                pnode, doc, store, tname, child_path, child_required, columns
            )
            if plan is not None:
                children.append((pname, plan))
        return _ObjectPlan(required=required, children=children)

    pname = prop_path.rsplit(".", 1)[-1] if prop_path else prop_path
    if pname not in columns:
        if required:
            raise DocumentError(
                f"tables.{tname}.format: schema property {prop_path!r} has no matching column"
            )
        return None
    return _LeafPlan(column=pname, required=required, type=_leaf_type(node))


def _compile_json_record_schema(
    fmt: FormatSpec, base_dir: Path | None, tname: str, columns: set[str]
) -> "_LeafPlan | _ObjectPlan":
    store: dict[str, Any] = {}
    primary_doc: Any = None
    for i, s in enumerate(fmt.schemas):
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

    return _compile_json_node(record_node, primary_doc, store, tname, "", True, columns)


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
    plan: "_LeafPlan | _ObjectPlan", row: dict[str, Any]
) -> tuple[bool, Any]:
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


def _render_json_shaped(metadata: Metadata, t: TableSpec, glob: str, path: Path) -> None:
    col_names, rows = _ordered_rows(glob, t.primary_key)
    plan = _compile_json_record_schema(t.format, metadata.base_dir, t.name, set(col_names))
    records = []
    for row in rows:
        row_dict = dict(zip(col_names, row))
        _, record = _shape_json_node(plan, row_dict)
        records.append(record)

    if t.format.kind == "json":
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
            f.write("\n")
    else:
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, separators=(",", ":")))
                f.write("\n")


# --------------------------------------------------------------------------
# xml, no schema
# --------------------------------------------------------------------------


def _render_xml_plain(t: TableSpec, glob: str, path: Path) -> None:
    fmt = t.format
    root_name = fmt.root or t.name
    record_name = fmt.record or _singular(t.name)

    col_names, rows = _ordered_rows(glob, t.primary_key)

    root_el = ET.Element(root_name)
    for row in rows:
        rec_el = ET.SubElement(root_el, record_name)
        for cname, value in zip(col_names, row):
            if value is None:
                continue
            child = ET.SubElement(rec_el, cname)
            child.text = _xml_text(value)

    _write_xml_tree(root_el, path)


def _write_xml_tree(root_el: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root_el)
    ET.indent(tree, space="  ")
    tree.write(str(path), xml_declaration=True, encoding="unicode")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n")


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


def _compile_xsd_sequence(
    ct: ET.Element,
    elements: dict[str, ET.Element],
    complex_types: dict[str, ET.Element],
    tname: str,
    columns: set[str],
) -> list["_XLeafPlan | _XObjectPlan"]:
    seq = ct.find(f"{_XS}sequence")
    if seq is None:
        seq = ct.find(f"{_XS}all")
    if seq is None:
        return []
    children: list[_XLeafPlan | _XObjectPlan] = []
    for particle_el in seq.findall(f"{_XS}element"):
        plan = _compile_xsd_particle(particle_el, elements, complex_types, tname, columns)
        if plan is not None:
            children.append(plan)
    return children


def _compile_xsd_particle(
    particle_el: ET.Element,
    elements: dict[str, ET.Element],
    complex_types: dict[str, ET.Element],
    tname: str,
    columns: set[str],
) -> "_XLeafPlan | _XObjectPlan | None":
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
    ct = _xsd_complex_type_of(type_source, complex_types)

    if ct is None:
        if name not in columns:
            if required:
                raise DocumentError(
                    f"tables.{tname}.format: schema element {name!r} has no matching column"
                )
            return None
        return _XLeafPlan(name=name, column=name, required=required)

    children = _compile_xsd_sequence(ct, elements, complex_types, tname, columns)
    return _XObjectPlan(name=name, required=required, children=children)


def _compile_xsd_record(
    record_el: ET.Element,
    elements: dict[str, ET.Element],
    complex_types: dict[str, ET.Element],
    tname: str,
    columns: set[str],
) -> _XObjectPlan:
    name = record_el.get("name")
    ct = _xsd_complex_type_of(record_el, complex_types)
    if ct is None:
        raise DocumentError(
            f"tables.{tname}.format: record element {name!r} has no complex content to shape"
        )
    children = _compile_xsd_sequence(ct, elements, complex_types, tname, columns)
    return _XObjectPlan(name=name, required=True, children=children)


def _render_xsd_node(plan: "_XLeafPlan | _XObjectPlan", row: dict[str, Any]) -> ET.Element | None:
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
        if child_el is not None:
            el.append(child_el)
            any_child = True
    if not any_child and not plan.required:
        return None
    return el


def _render_xml_shaped(metadata: Metadata, t: TableSpec, glob: str, path: Path) -> None:
    fmt = t.format
    roots = _load_xsd_documents(fmt, metadata.base_dir, t.name)
    elements, complex_types = _collect_xsd_globals(roots)
    record_el, wrapper_name = _find_record_element(fmt, t, elements, complex_types)
    root_name = fmt.root or wrapper_name or t.name

    col_names, rows = _ordered_rows(glob, t.primary_key)
    plan = _compile_xsd_record(record_el, elements, complex_types, t.name, set(col_names))

    root_el = ET.Element(root_name)
    for row in rows:
        row_dict = dict(zip(col_names, row))
        rec_el = _render_xsd_node(plan, row_dict)
        if rec_el is not None:
            root_el.append(rec_el)

    _write_xml_tree(root_el, path)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def _render_document(metadata: Metadata, t: TableSpec, glob: str, path: Path) -> None:
    fmt = t.format
    if fmt.kind in ("json", "jsonl"):
        if fmt.schemas:
            _render_json_shaped(metadata, t, glob, path)
        else:
            _render_json_plain(t, glob, path)
    else:  # xml
        if fmt.schemas:
            _render_xml_shaped(metadata, t, glob, path)
        else:
            _render_xml_plain(t, glob, path)


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
        glob = backbone.table_glob(tname, t.source)
        _render_document(metadata, t, glob, path)
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
                    tree = ET.parse(path)
                    n = len(list(tree.getroot()))
                except ET.ParseError as e:
                    violations.append(
                        f"{tname}: document {path} could not be parsed as XML: {e}"
                    )
                    continue

            if n != expected:
                violations.append(
                    f"{tname}: document {path} has {n} records, expected {expected}"
                )
    finally:
        con.close()

    return violations
