"""Acceptance tests for verisynth.metadata (Metadata DSL)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from verisynth.metadata import MetadataError, load_metadata, parse_metadata

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PATH = REPO_ROOT / "examples" / "retail.yaml"


def _raw_doc() -> dict:
    with open(EXAMPLE_PATH) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------
# 1. load_metadata succeeds; spot-check parsed values
# --------------------------------------------------------------------------


def test_load_metadata_example():
    md = load_metadata("examples/retail.yaml")

    assert md.version == 1
    assert md.seed == 42

    customers = md.tables["customers"]
    assert customers.role == "root"
    assert customers.rows == 10000
    assert customers.primary_key == "customer_id"

    orders = md.tables["orders"]
    assert orders.role == "child"
    assert orders.parent == "customers"
    assert orders.cardinality.kind == "poisson"
    assert orders.child_stride == 64

    # copula
    assert len(customers.copulas) == 1
    copula = customers.copulas[0]
    assert copula.name == "profile"
    assert copula.columns == ["age", "income"]
    assert copula.correlation == [[1.0, 0.55], [0.55, 1.0]]

    # temporal anchors
    assert orders.columns["ordered_at"].temporal.anchor == "customers.signup_at"
    assert orders.columns["shipped_at"].temporal.anchor == "ordered_at"

    # derived expr
    assert len(orders.derived) == 1
    assert orders.derived[0].name == "order_total_eur"
    assert orders.derived[0].expr == "order_total * 0.92"


# --------------------------------------------------------------------------
# 2. table_order()
# --------------------------------------------------------------------------


def test_table_order_customers_before_orders():
    md = load_metadata("examples/retail.yaml")
    order = md.table_order()
    assert "customers" in order
    assert "orders" in order
    assert order.index("customers") < order.index("orders")


# --------------------------------------------------------------------------
# 3. Invalid documents
# --------------------------------------------------------------------------


def _unknown_dist_kind(doc):
    doc["tables"]["customers"]["columns"]["region"]["distribution"]["kind"] = "bogus"
    return doc, "customers.columns.region.distribution"


def _probs_not_summing(doc):
    doc["tables"]["customers"]["columns"]["region"]["distribution"]["probs"] = [0.5, 0.3, 0.3]
    return doc, "customers.columns.region.distribution"


def _child_without_cardinality(doc):
    del doc["tables"]["orders"]["cardinality"]
    return doc, "orders.cardinality"


def _pk_missing_column(doc):
    doc["tables"]["customers"]["primary_key"] = "does_not_exist"
    return doc, "customers.primary_key"


def _correlation_not_symmetric(doc):
    doc["tables"]["customers"]["copulas"][0]["correlation"] = [[1.0, 0.55], [0.3, 1.0]]
    return doc, "customers.copulas"


def _temporal_cycle(doc):
    # ordered_at now anchors on shipped_at (same table), and shipped_at already
    # anchors on ordered_at -> mutual cycle.
    doc["tables"]["orders"]["columns"]["ordered_at"]["temporal"]["anchor"] = "shipped_at"
    return doc, "orders.columns"


def _cardinality_max_ge_stride(doc):
    doc["tables"]["orders"]["cardinality"]["max"] = 64  # == child_stride
    return doc, "orders.cardinality"


def _both_generator_and_distribution(doc):
    doc["tables"]["customers"]["columns"]["customer_id"]["distribution"] = {
        "kind": "normal",
        "mean": 1.0,
        "std": 1.0,
    }
    return doc, "customers.columns.customer_id"


INVALID_CASES = [
    _unknown_dist_kind,
    _probs_not_summing,
    _child_without_cardinality,
    _pk_missing_column,
    _correlation_not_symmetric,
    _temporal_cycle,
    _cardinality_max_ge_stride,
    _both_generator_and_distribution,
]


@pytest.mark.parametrize("mutator", INVALID_CASES, ids=[c.__name__ for c in INVALID_CASES])
def test_invalid_documents_raise_with_path(mutator):
    doc = copy.deepcopy(_raw_doc())
    doc, expected_substring = mutator(doc)
    with pytest.raises(MetadataError) as excinfo:
        parse_metadata(doc)
    assert expected_substring in str(excinfo.value)
