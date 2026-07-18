"""Tests for `verisynth explain` (TASK CARD 15).

See docs/ARCHITECTURE.md §2-§5, §8.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from verisynth.explain import describe_distribution, explain_metadata, humanize_seconds
from verisynth.metadata import (
    ColumnSpec,
    DistributionSpec,
    Metadata,
    TableSpec,
    load_metadata,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RETAIL_YAML = REPO_ROOT / "examples" / "retail.yaml"
OLIST_METADATA = REPO_ROOT / "examples" / "olist" / "metadata.olist.yaml"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "verisynth.cli", *args],
        capture_output=True,
        text=True,
    )


# --------------------------------------------------------------------------
# 1. humanize_seconds
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (45, "45s"),
        (300, "5 min"),
        (7200, "2 h"),
        (300000, "3.5 days"),
    ],
)
def test_humanize_seconds(seconds, expected):
    assert humanize_seconds(seconds) == expected


def test_humanize_seconds_boundaries():
    assert humanize_seconds(89) == "89s"
    assert humanize_seconds(90) == "1.5 min"
    assert humanize_seconds(89 * 60) == "89 min"
    assert humanize_seconds(36 * 3600) == "1.5 days"


# --------------------------------------------------------------------------
# 2. retail example
# --------------------------------------------------------------------------


def test_retail_explain_contents():
    md = load_metadata(RETAIL_YAML)
    doc = explain_metadata(md)

    assert "customers" in doc
    assert "orders" in doc
    assert "Root entity, 10,000 rows" in doc
    assert "0.55" in doc
    assert "Event flow:" in doc
    assert "order_total_eur" in doc
    assert "order_total * 0.92" in doc
    assert not doc.startswith("\n")
    assert not any(line != line.rstrip() for line in doc.splitlines())


# --------------------------------------------------------------------------
# 3. Olist fitted metadata
# --------------------------------------------------------------------------


def test_olist_explain_contents():
    md = load_metadata(OLIST_METADATA)
    doc = explain_metadata(md)

    assert "## Source: crm" in doc
    assert "## Source: shop" in doc
    assert "## Source: inventory" in doc
    assert "inherited from `crm_contacts.state`" in doc
    assert "98%" in doc  # inv_shipments bernoulli
    assert "60%" in doc  # customers bernoulli
    assert "reference into `inv_products`" in doc

    for tname in md.tables:
        assert tname in doc

    assert "None" not in doc


# --------------------------------------------------------------------------
# 4. CLI
# --------------------------------------------------------------------------


def test_cli_explain_stdout():
    result = _run("explain", "-m", str(OLIST_METADATA))
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("# Synthetic dataset:")


def test_cli_explain_to_file(tmp_path):
    out_path = tmp_path / "explain.md"
    result = _run("explain", "-m", str(OLIST_METADATA), "-o", str(out_path))
    assert result.returncode == 0, result.stderr
    assert f"wrote {out_path}" in result.stdout
    assert out_path.exists()
    text = out_path.read_text()
    assert text.startswith("# Synthetic dataset:")


# --------------------------------------------------------------------------
# 5. Robustness against unrecognized distribution kinds
# --------------------------------------------------------------------------


def test_describe_distribution_unknown_kind_falls_back():
    spec = DistributionSpec(kind="not_a_real_kind", params={"foo": 1, "bar": 2})
    text = describe_distribution(spec)
    assert "not_a_real_kind" in text


def test_explain_metadata_does_not_crash_on_unexpected_kind():
    weird_dist = DistributionSpec(kind="totally_bogus", params={"x": 1})
    col_a = ColumnSpec(name="id", type="int64", generator="key")
    col_b = ColumnSpec(name="weird", type="float64", distribution=weird_dist)
    table = TableSpec(
        name="widgets",
        role="root",
        columns={"id": col_a, "weird": col_b},
        primary_key="id",
        rows=10,
    )
    md = Metadata(version=1, seed=0, tables={"widgets": table})

    doc = explain_metadata(md)  # must not raise
    assert "widgets" in doc
    assert "totally_bogus" in doc
