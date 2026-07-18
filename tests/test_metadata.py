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


# --------------------------------------------------------------------------
# 4. generator: parent:{column} (master-data inheritance)
# --------------------------------------------------------------------------


def _two_table_doc_with_inherited_state() -> dict:
    """A minimal root+child document where the child inherits 'state' from
    its parent via generator: parent:state."""
    return {
        "version": 1,
        "seed": 1,
        "tables": {
            "crm_contacts": {
                "role": "root",
                "rows": 100,
                "primary_key": "contact_id",
                "source": "crm",
                "columns": {
                    "contact_id": {"type": "int64", "generator": "key"},
                    "state": {
                        "type": "string",
                        "distribution": {
                            "kind": "categorical",
                            "categories": ["A", "B", "C"],
                            "probs": [0.5, 0.3, 0.2],
                        },
                    },
                },
                "derived": [{"name": "state_lower", "expr": "lower(state)"}],
            },
            "customers": {
                "role": "child",
                "parent": "crm_contacts",
                "cardinality": {"kind": "bernoulli", "p": 0.6},
                "child_stride": 2,
                "primary_key": "customer_id",
                "source": "shop",
                "columns": {
                    "customer_id": {"type": "int64", "generator": "key"},
                    "contact_id": {"type": "int64", "generator": "parent_key"},
                    "state": {"type": "string", "generator": "parent:state"},
                },
            },
        },
    }


def test_generator_parent_column_valid():
    doc = _two_table_doc_with_inherited_state()
    md = parse_metadata(doc)
    assert md.tables["customers"].columns["state"].generator == "parent:state"
    assert md.tables["crm_contacts"].source == "crm"
    assert md.tables["customers"].source == "shop"


def test_generator_parent_on_root_table_invalid():
    doc = _two_table_doc_with_inherited_state()
    # crm_contacts is a root table: 'parent:{column}' is only valid on child
    # tables. Add a second root column so we don't collide with the PK.
    doc["tables"]["crm_contacts"]["columns"]["extra"] = {"type": "string", "generator": "parent:state"}
    with pytest.raises(MetadataError) as excinfo:
        parse_metadata(doc)
    assert "crm_contacts.columns.extra.generator" in str(excinfo.value)


def test_generator_parent_nonexistent_column_invalid():
    doc = _two_table_doc_with_inherited_state()
    doc["tables"]["customers"]["columns"]["state"]["generator"] = "parent:does_not_exist"
    with pytest.raises(MetadataError) as excinfo:
        parse_metadata(doc)
    assert "customers.columns.state.generator" in str(excinfo.value)


def test_generator_parent_type_mismatch_invalid():
    doc = _two_table_doc_with_inherited_state()
    doc["tables"]["customers"]["columns"]["state"]["type"] = "int64"
    with pytest.raises(MetadataError) as excinfo:
        parse_metadata(doc)
    assert "customers.columns.state.generator" in str(excinfo.value)


def test_generator_parent_references_derived_column_invalid():
    doc = _two_table_doc_with_inherited_state()
    doc["tables"]["customers"]["columns"]["state"]["generator"] = "parent:state_lower"
    with pytest.raises(MetadataError) as excinfo:
        parse_metadata(doc)
    assert "customers.columns.state.generator" in str(excinfo.value)


# --------------------------------------------------------------------------
# 5. bernoulli{p} cardinality
# --------------------------------------------------------------------------


def test_bernoulli_cardinality_p_out_of_range_invalid():
    doc = _two_table_doc_with_inherited_state()
    doc["tables"]["customers"]["cardinality"]["p"] = 1.5
    with pytest.raises(MetadataError) as excinfo:
        parse_metadata(doc)
    assert "customers.cardinality" in str(excinfo.value)


def test_bernoulli_cardinality_valid_with_stride_2():
    doc = _two_table_doc_with_inherited_state()
    doc["tables"]["customers"]["cardinality"]["p"] = 0.6
    doc["tables"]["customers"]["child_stride"] = 2
    md = parse_metadata(doc)
    assert md.tables["customers"].cardinality.kind == "bernoulli"
    assert md.tables["customers"].cardinality.params["p"] == 0.6
    assert md.tables["customers"].child_stride == 2


def test_bernoulli_cardinality_stride_1_invalid():
    doc = _two_table_doc_with_inherited_state()
    doc["tables"]["customers"]["child_stride"] = 1
    with pytest.raises(MetadataError) as excinfo:
        parse_metadata(doc)
    assert "customers.cardinality" in str(excinfo.value)


# --------------------------------------------------------------------------
# 6. source: round-trip
# --------------------------------------------------------------------------


def test_source_round_trips_through_metadata_to_dict():
    from verisynth.metadata import metadata_to_dict

    doc = _two_table_doc_with_inherited_state()
    md = parse_metadata(doc)
    d = metadata_to_dict(md)
    assert d["tables"]["crm_contacts"]["source"] == "crm"
    assert d["tables"]["customers"]["source"] == "shop"

    reparsed = parse_metadata(d)
    assert metadata_to_dict(reparsed) == d


def test_source_omitted_when_none():
    from verisynth.metadata import metadata_to_dict

    md = load_metadata(EXAMPLE_PATH)
    d = metadata_to_dict(md)
    assert "source" not in d["tables"]["customers"]
    assert "source" not in d["tables"]["orders"]


# --------------------------------------------------------------------------
# 7. Categorical category-value coercion (int / bool categories survive
#    load -> dict round-trip unchanged in type).
# --------------------------------------------------------------------------


def test_int_categories_survive_round_trip_as_ints():
    from verisynth.metadata import metadata_to_dict

    doc = {
        "version": 1,
        "seed": 1,
        "tables": {
            "t": {
                "role": "root",
                "rows": 5,
                "primary_key": "id",
                "columns": {
                    "id": {"type": "int64", "generator": "key"},
                    "score": {
                        "type": "int64",
                        "distribution": {
                            "kind": "categorical",
                            "categories": [1, 2, 3],
                            "probs": [0.2, 0.3, 0.5],
                        },
                    },
                },
            }
        },
    }
    md = parse_metadata(doc)
    categories = md.tables["t"].columns["score"].distribution.params["categories"]
    assert categories == [1, 2, 3]
    assert all(isinstance(c, int) and not isinstance(c, bool) for c in categories)

    d = metadata_to_dict(md)
    assert d["tables"]["t"]["columns"]["score"]["distribution"]["categories"] == [1, 2, 3]
    assert all(
        isinstance(c, int) and not isinstance(c, bool)
        for c in d["tables"]["t"]["columns"]["score"]["distribution"]["categories"]
    )

    reparsed = parse_metadata(d)
    assert metadata_to_dict(reparsed) == d


def test_bool_categories_survive_round_trip_as_bools():
    from verisynth.metadata import metadata_to_dict

    doc = {
        "version": 1,
        "seed": 1,
        "tables": {
            "t": {
                "role": "root",
                "rows": 5,
                "primary_key": "id",
                "columns": {
                    "id": {"type": "int64", "generator": "key"},
                    "flag": {
                        "type": "bool",
                        "distribution": {
                            "kind": "categorical",
                            "categories": [True, False],
                            "probs": [0.7, 0.3],
                        },
                    },
                },
            }
        },
    }
    md = parse_metadata(doc)
    categories = md.tables["t"].columns["flag"].distribution.params["categories"]
    assert categories == [True, False]
    assert all(isinstance(c, bool) for c in categories)

    d = metadata_to_dict(md)
    assert d["tables"]["t"]["columns"]["flag"]["distribution"]["categories"] == [True, False]
    assert all(
        isinstance(c, bool)
        for c in d["tables"]["t"]["columns"]["flag"]["distribution"]["categories"]
    )


# --------------------------------------------------------------------------
# 8. `zipf{a, n}` distribution kind + `reference:` dimension column
#    (TASK CARD 13, docs/ARCHITECTURE.md §2)
# --------------------------------------------------------------------------


def _dimref_doc() -> dict:
    """root products{rows: 100} + shops -> sales(child) with a fact column
    `product_ref` that references `products` via a zipf popularity
    distribution."""
    return {
        "version": 1,
        "seed": 1,
        "tables": {
            "products": {
                "role": "root",
                "rows": 100,
                "primary_key": "product_id",
                "columns": {
                    "product_id": {"type": "int64", "generator": "key"},
                },
            },
            "shops": {
                "role": "root",
                "rows": 20,
                "primary_key": "shop_id",
                "columns": {
                    "shop_id": {"type": "int64", "generator": "key"},
                },
            },
            "sales": {
                "role": "child",
                "parent": "shops",
                "cardinality": {"kind": "poisson", "lam": 2.0, "max": 15},
                "child_stride": 16,
                "primary_key": "sale_id",
                "columns": {
                    "sale_id": {"type": "int64", "generator": "key"},
                    "shop_id": {"type": "int64", "generator": "parent_key"},
                    "product_ref": {
                        "type": "int64",
                        "distribution": {"kind": "zipf", "a": 1.3, "n": 100},
                        "reference": "products",
                    },
                },
            },
        },
    }


def test_dimension_reference_valid_parses_and_round_trips():
    from verisynth.metadata import metadata_to_dict

    doc = _dimref_doc()
    md = parse_metadata(doc)

    product_ref = md.tables["sales"].columns["product_ref"]
    assert product_ref.reference == "products"
    assert product_ref.distribution.kind == "zipf"
    assert product_ref.distribution.params["a"] == 1.3
    assert product_ref.distribution.params["n"] == 100
    assert isinstance(product_ref.distribution.params["n"], int)

    d = metadata_to_dict(md)
    assert d["tables"]["sales"]["columns"]["product_ref"]["reference"] == "products"
    assert d["tables"]["sales"]["columns"]["product_ref"]["distribution"] == {
        "kind": "zipf",
        "a": 1.3,
        "n": 100,
    }

    reparsed = parse_metadata(d)
    assert metadata_to_dict(reparsed) == d


def test_dimension_reference_to_nonexistent_table_invalid():
    doc = _dimref_doc()
    doc["tables"]["sales"]["columns"]["product_ref"]["reference"] = "does_not_exist"
    with pytest.raises(MetadataError) as excinfo:
        parse_metadata(doc)
    assert "sales.columns.product_ref.reference" in str(excinfo.value)


def test_dimension_reference_to_child_table_invalid():
    doc = _dimref_doc()
    # 'sales' is itself a child table -> referenced table must have role root.
    doc["tables"]["sales"]["columns"]["product_ref"]["reference"] = "sales"
    with pytest.raises(MetadataError) as excinfo:
        parse_metadata(doc)
    assert "sales.columns.product_ref.reference" in str(excinfo.value)


def test_dimension_reference_on_string_column_invalid():
    doc = _dimref_doc()
    doc["tables"]["sales"]["columns"]["product_ref"]["type"] = "string"
    with pytest.raises(MetadataError) as excinfo:
        parse_metadata(doc)
    assert "sales.columns.product_ref.reference" in str(excinfo.value)


def test_dimension_reference_on_generator_column_invalid():
    doc = _dimref_doc()
    # shop_id is a generator column (parent_key); it has no distribution, so a
    # reference on it must be rejected.
    doc["tables"]["sales"]["columns"]["shop_id"]["reference"] = "products"
    with pytest.raises(MetadataError) as excinfo:
        parse_metadata(doc)
    assert "sales.columns.shop_id.reference" in str(excinfo.value)


def test_zipf_distribution_a_le_1_invalid():
    doc = _dimref_doc()
    doc["tables"]["sales"]["columns"]["product_ref"]["distribution"]["a"] = 1.0
    with pytest.raises(MetadataError) as excinfo:
        parse_metadata(doc)
    assert "sales.columns.product_ref.distribution" in str(excinfo.value)
