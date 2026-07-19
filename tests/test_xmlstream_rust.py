"""Parity tests for the Rust streaming-XML functions in `verisynth_kernels`.

See docs/ARCHITECTURE.md §12 and `verisynth/xmlstream.py` (the normative
record-flattening spec). Skips entirely when the compiled wheel isn't
installed, or predates the XML streaming functions, so the suite stays green
on machines that never built the extension.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

verisynth_kernels = pytest.importorskip("verisynth_kernels")

if not (
    hasattr(verisynth_kernels, "stream_xml_file")
    and hasattr(verisynth_kernels, "count_xml_records")
):
    pytest.skip(
        "installed verisynth_kernels wheel predates XML streaming functions",
        allow_module_level=True,
    )

from verisynth import xmlstream as xs  # noqa: E402


# --------------------------------------------------------------------------
# Helpers: run both backends over the same file and normalize to plain lists
# for comparison, bypassing the BACKEND-selecting dispatch layer entirely.
# --------------------------------------------------------------------------


def _reference_batches(path, batch_rows):
    """[(names, columns, n_rows), ...] from the pure-Python reference."""
    out = []
    for df in xs._iter_batches_reference(path, batch_rows):
        names = list(df.columns)
        columns = [df[c].to_list() for c in names]
        out.append((names, columns, df.height))
    return out


def _rust_batches(path, batch_rows):
    """[(names, columns, n_rows), ...] from the Rust `stream_xml_file`."""
    out = []
    for names, columns in verisynth_kernels.stream_xml_file(str(path), batch_rows):
        names = list(names)
        columns = [list(c) for c in columns]
        n_rows = len(columns[0]) if columns else None
        out.append((names, columns, n_rows))
    return out


def _assert_batches_match(path, batch_rows):
    ref = _reference_batches(path, batch_rows)
    rust = _rust_batches(path, batch_rows)
    assert len(ref) == len(rust), f"batch count mismatch: {len(ref)} vs {len(rust)}"
    for i, ((rn, rc, rrows), (un, uc, urows)) in enumerate(zip(ref, rust)):
        assert rn == un, f"batch {i}: column names mismatch: {rn!r} vs {un!r}"
        assert rc == uc, f"batch {i}: column values mismatch: {rc!r} vs {uc!r}"
        if urows is not None:
            assert rrows == urows, f"batch {i}: row count mismatch: {rrows} vs {urows}"
    return ref, rust


# --------------------------------------------------------------------------
# Case 1: simple flat records, some columns missing per record -> None,
# multiple batches.
# --------------------------------------------------------------------------


def test_flat_records_multi_batch_parity(tmp_path):
    path = tmp_path / "flat.xml"
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<root>
  <record><id>1</id><name>Alice</name><age>30</age></record>
  <record><id>2</id><name>Bob</name></record>
  <record><id>3</id><age>40</age></record>
  <record><id>4</id><name>Dana</name><age>22</age></record>
  <record><id>5</id><name>Eve</name></record>
</root>
""",
        encoding="utf-8",
    )
    ref, rust = _assert_batches_match(path, batch_rows=2)
    assert [r[2] for r in ref] == [2, 2, 1]
    assert [r[2] for r in rust] == [2, 2, 1]
    # sanity: the reference itself has the expected None-filled shape.
    names, columns, _ = ref[0]
    assert names == ["id", "name", "age"]
    assert columns[0] == ["1", "2"]
    assert columns[1] == ["Alice", "Bob"]
    names2, columns2, _ = ref[1]
    # batch 2 = records 3 ({id, age}) and 4 ({id, name, age}): first-seen
    # order across the batch puts "age" before "name".
    assert names2 == ["id", "age", "name"]
    assert columns2[2] == [None, "Dana"]


# --------------------------------------------------------------------------
# Case 2: nested containers (2 levels deep) with key collisions, plus
# duplicate direct leaves with the same tag.
# --------------------------------------------------------------------------


def test_nested_and_collision_parity(tmp_path):
    path = tmp_path / "nested.xml"
    path.write_text(
        """<root>
  <record>
    <id>1</id>
    <address>
      <id>99</id>
      <city>Springfield</city>
      <geo><id>7</id><lat>1.23</lat></geo>
    </address>
  </record>
  <record>
    <name>Bob</name>
    <name>Bobby</name>
  </record>
</root>
""",
        encoding="utf-8",
    )
    ref, rust = _assert_batches_match(path, batch_rows=100)
    assert len(ref) == 1
    names, columns, n_rows = ref[0]
    assert n_rows == 2
    row0 = dict(zip(names, [columns[i][0] for i in range(len(names))]))
    row1 = dict(zip(names, [columns[i][1] for i in range(len(names))]))
    assert row0 == {
        "id": "1",
        "address_id": "99",
        "city": "Springfield",
        "geo_id": "7",
        "lat": "1.23",
        "name": None,
        "record_name": None,
    }
    assert row1["name"] == "Bob"
    assert row1["record_name"] == "Bobby"


# --------------------------------------------------------------------------
# Case 3: namespaces - default xmlns and a prefixed tag -> local names in
# both backends.
# --------------------------------------------------------------------------


def test_namespace_local_names_parity(tmp_path):
    path = tmp_path / "ns.xml"
    path.write_text(
        """<root xmlns="http://example.com/default" xmlns:ns="http://example.com/ns">
  <record>
    <id>1</id>
    <ns:value>42</ns:value>
  </record>
</root>
""",
        encoding="utf-8",
    )
    ref, rust = _assert_batches_match(path, batch_rows=100)
    names, columns, _ = ref[0]
    assert names == ["id", "value"]
    assert columns[0] == ["1"]
    assert columns[1] == ["42"]


# --------------------------------------------------------------------------
# Case 4: entities, CDATA, whitespace-only text -> None, empty element -> None.
# --------------------------------------------------------------------------


def test_entities_cdata_whitespace_empty_parity(tmp_path):
    path = tmp_path / "text.xml"
    path.write_text(
        """<root>
  <record>
    <a>Tom &amp; Jerry</a>
    <b>x &lt; y</b>
    <c><![CDATA[<raw> & text]]></c>
    <d>   </d>
    <e/>
  </record>
</root>
""",
        encoding="utf-8",
    )
    ref, rust = _assert_batches_match(path, batch_rows=100)
    names, columns, _ = ref[0]
    assert names == ["a", "b", "c", "d", "e"]
    assert columns[0] == ["Tom & Jerry"]
    assert columns[1] == ["x < y"]
    assert columns[2] == ["<raw> & text"]
    assert columns[3] == [None]
    assert columns[4] == [None]


# --------------------------------------------------------------------------
# Case 5: count_xml_records parity, including an empty root.
# --------------------------------------------------------------------------


def test_count_xml_records_parity(tmp_path):
    files = {
        "flat.xml": """<root>
  <record><id>1</id></record>
  <record><id>2</id></record>
  <record><id>3</id></record>
</root>
""",
        "self_closed_empty_root.xml": "<root/>",
        "empty_root.xml": "<root></root>",
        "one_record.xml": "<root><record><id>1</id></record></root>",
    }
    for fname, content in files.items():
        path = tmp_path / fname
        path.write_text(content, encoding="utf-8")
        ref_count = xs._count_records_reference(path)
        rust_count = int(verisynth_kernels.count_xml_records(str(path)))
        assert rust_count == ref_count, f"{fname}: rust={rust_count} ref={ref_count}"

    assert int(verisynth_kernels.count_xml_records(str(tmp_path / "self_closed_empty_root.xml"))) == 0
    assert int(verisynth_kernels.count_xml_records(str(tmp_path / "empty_root.xml"))) == 0


def test_count_xml_records_parity_on_nested_and_text_files(tmp_path):
    # Re-use the more elaborate fixtures from the other parity tests to
    # exercise count_xml_records on nested/namespaced/entity-laden documents.
    nested = tmp_path / "nested.xml"
    nested.write_text(
        """<root>
  <record>
    <id>1</id>
    <address><id>99</id><city>Springfield</city><geo><id>7</id><lat>1.23</lat></geo></address>
  </record>
  <record><name>Bob</name><name>Bobby</name></record>
</root>
""",
        encoding="utf-8",
    )
    ns = tmp_path / "ns.xml"
    ns.write_text(
        """<root xmlns="http://example.com/default" xmlns:ns="http://example.com/ns">
  <record><id>1</id><ns:value>42</ns:value></record>
</root>
""",
        encoding="utf-8",
    )
    text = tmp_path / "text.xml"
    text.write_text(
        """<root>
  <record><a>Tom &amp; Jerry</a><b>x &lt; y</b><c><![CDATA[<raw> & text]]></c><d>   </d><e/></record>
</root>
""",
        encoding="utf-8",
    )
    expected = {nested: 2, ns: 1, text: 1}
    for path, want in expected.items():
        ref_count = xs._count_records_reference(path)
        rust_count = int(verisynth_kernels.count_xml_records(str(path)))
        assert rust_count == ref_count == want, f"{path.name}: rust={rust_count} ref={ref_count}"


# --------------------------------------------------------------------------
# Case 6: malformed XML raises. Only the rust error type/message is asserted;
# the reference (ET) is checked separately since it raises a different type.
# --------------------------------------------------------------------------


def test_malformed_xml_raises_value_error_with_path(tmp_path):
    path = tmp_path / "bad.xml"
    path.write_text(
        "<root><record><id>1</id></root>",  # missing </record>
        encoding="utf-8",
    )

    with pytest.raises(ET.ParseError):
        list(xs._iter_records_reference(path))

    it = verisynth_kernels.stream_xml_file(str(path), 10)
    with pytest.raises(ValueError, match=re.escape(str(path))):
        next(it)

    with pytest.raises(ValueError, match=re.escape(str(path))):
        verisynth_kernels.count_xml_records(str(path))
