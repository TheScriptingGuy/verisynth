"""Integration tests for the Olist example (TASK CARD 9, extended by
TASK CARD 12 for the two-source CRM/shop dataset, and TASK CARD 14 for the
third "inventory" source: master product catalog + shipment orders).

Exercises the committed `examples/olist` sample end-to-end: skeleton/fitted
metadata loading, fit reproducibility against the committed
`metadata.olist.yaml` "expected output" artifact, generation + validation,
statistical fidelity vs. the committed source sample, cross-source
master-data consistency (CRM as customer master; inventory as product
master + shipment orders as children of shop orders), and determinism.

The committed Parquet sample under `examples/olist/data/` is small (15k
shop customers, 25k CRM contacts, ~9.4k inventory products) and always
present, so these tests always run (no skip marker).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import yaml
from scipy import stats

from verisynth.backbone import ParquetBackbone, validate_dataset
from verisynth.engine import Engine
from verisynth.fit import fit_metadata
from verisynth.metadata import load_metadata, metadata_to_dict

OLIST_DIR = Path(__file__).resolve().parent.parent / "examples" / "olist"
SKELETON_PATH = OLIST_DIR / "skeleton.yaml"
FITTED_PATH = OLIST_DIR / "metadata.olist.yaml"
DATA_DIR = OLIST_DIR / "data"

# All 9 tables across the three source systems (crm: crm_contacts,
# crm_tickets; shop: customers, orders, order_items, order_payments,
# order_reviews; inventory: inv_products, inv_shipments).
TABLE_NAMES = (
    "crm_contacts",
    "crm_tickets",
    "customers",
    "orders",
    "order_items",
    "order_payments",
    "order_reviews",
    "inv_products",
    "inv_shipments",
)

SOURCE_TABLES = (
    ("crm", "crm_contacts"),
    ("crm", "crm_tickets"),
    ("shop", "customers"),
    ("shop", "orders"),
    ("shop", "order_items"),
    ("shop", "order_payments"),
    ("shop", "order_reviews"),
    ("inventory", "inv_products"),
    ("inventory", "inv_shipments"),
)


def _load_source_frames() -> dict[str, pl.DataFrame]:
    return {tname: pl.read_parquet(DATA_DIR / f"{tname}.parquet") for tname in TABLE_NAMES}


def _load_synth_frames(out_dir: Path, metadata) -> dict[str, pl.DataFrame]:
    backbone = ParquetBackbone(out_dir)
    frames = {}
    for tname in TABLE_NAMES:
        glob = backbone.table_glob(tname, metadata.tables[tname].source)
        frames[tname] = pl.read_parquet(glob)
    return frames


def _load_crm_prep_module():
    """Load ``examples/olist/prepare_crm_data.py`` by path (``examples/``
    is not a package) to import its authored ground-truth constants."""
    path = OLIST_DIR / "prepare_crm_data.py"
    spec = importlib.util.spec_from_file_location("olist_prepare_crm_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_inventory_prep_module():
    """Load ``examples/olist/prepare_inventory_data.py`` by path (mirrors
    ``_load_crm_prep_module``) to import its authored ground-truth
    constants (TASK CARD 14)."""
    path = OLIST_DIR / "prepare_inventory_data.py"
    spec = importlib.util.spec_from_file_location("olist_prepare_inventory_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Recursive numeric-tolerant dict comparator.
# --------------------------------------------------------------------------


def _assert_close(a, b, path: str = "$", rel: float = 1e-6, abs_tol: float = 1e-9) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), f"{path}: keys differ ({sorted(a.keys())} vs {sorted(b.keys())})"
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}", rel, abs_tol)
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{path}: length differs ({len(a)} vs {len(b)})"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{path}[{i}]", rel, abs_tol)
    elif isinstance(a, bool) or isinstance(b, bool):
        assert a == b, f"{path}: {a!r} != {b!r}"
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        assert a == pytest.approx(b, rel=rel, abs=abs_tol), f"{path}: {a!r} != {b!r}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def _freq(series: pl.Series) -> dict:
    counts = series.value_counts()
    col = series.name
    total = counts["count"].sum()
    return dict(zip(counts[col].to_list(), (counts["count"] / total).to_list()))


def _dist_probs(metadata, table: str, column: str) -> dict:
    dist = metadata.tables[table].columns[column].distribution
    return dict(zip(dist.params["categories"], dist.params["probs"]))


# --------------------------------------------------------------------------
# Fixtures: fit + generate the committed sample once for the whole module.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def source_frames() -> dict[str, pl.DataFrame]:
    return _load_source_frames()


@pytest.fixture(scope="module")
def fitted_metadata():
    return load_metadata(FITTED_PATH)


@pytest.fixture(scope="module")
def crm_prep_module():
    return _load_crm_prep_module()


@pytest.fixture(scope="module")
def inventory_prep_module():
    return _load_inventory_prep_module()


@pytest.fixture(scope="module")
def synth_out_dir(tmp_path_factory, fitted_metadata) -> Path:
    out_dir = tmp_path_factory.mktemp("olist_gen")
    engine = Engine(fitted_metadata)
    engine.generate(str(out_dir), num_partitions=2)
    return out_dir


@pytest.fixture(scope="module")
def synth_frames(synth_out_dir, fitted_metadata) -> dict[str, pl.DataFrame]:
    return _load_synth_frames(synth_out_dir, fitted_metadata)


# --------------------------------------------------------------------------
# 1. Skeleton and fitted metadata load.
# --------------------------------------------------------------------------


def test_skeleton_and_fitted_metadata_load(fitted_metadata):
    skeleton = load_metadata(SKELETON_PATH)
    assert set(skeleton.tables) == set(TABLE_NAMES)

    review_dist = fitted_metadata.tables["order_reviews"].columns["review_score"].distribution
    assert review_dist.kind == "categorical"
    # metadata.py's generic YAML round-trip coercion (see metadata._coerce)
    # widens numeric distribution params to float on load, independent of
    # this task's fit.py change; the *values* are still integral 1..5.
    assert [int(c) for c in review_dist.params["categories"]] == [1, 2, 3, 4, 5]
    assert sum(review_dist.params["probs"]) == pytest.approx(1.0, abs=1e-6)

    # Source routing (TASK CARD 12 / docs/ARCHITECTURE.md §8): CRM tables
    # are the customer master, shop tables are downstream.
    assert skeleton.tables["crm_contacts"].source == "crm"
    assert skeleton.tables["crm_tickets"].source == "crm"
    for tname in ("customers", "orders", "order_items", "order_payments", "order_reviews"):
        assert skeleton.tables[tname].source == "shop"

    # customers is now a bernoulli child of crm_contacts, inheriting state.
    customers_t = skeleton.tables["customers"]
    assert customers_t.role == "child"
    assert customers_t.parent == "crm_contacts"
    assert customers_t.cardinality.kind == "bernoulli"
    assert customers_t.columns["customer_state"].generator == "parent:state"


def test_fit_produces_plain_int_categories_for_int64_categorical_columns(source_frames):
    """Direct check of the fit.py contract extension (TASK CARD 9 §1): the
    in-memory ``fit_metadata`` output for an int64 column declared
    categorical in the skeleton has plain Python int categories."""
    skeleton = load_metadata(SKELETON_PATH)
    fitted = fit_metadata(source_frames, skeleton)

    review_dist = fitted.tables["order_reviews"].columns["review_score"].distribution
    assert review_dist.kind == "categorical"
    assert review_dist.params["categories"] == [1, 2, 3, 4, 5]
    assert all(isinstance(c, int) for c in review_dist.params["categories"])

    installments_dist = fitted.tables["order_payments"].columns["payment_installments"].distribution
    assert installments_dist.kind == "categorical"
    assert all(isinstance(c, int) for c in installments_dist.params["categories"])
    assert installments_dist.params["categories"] == sorted(installments_dist.params["categories"])

    csat_dist = fitted.tables["crm_tickets"].columns["csat_score"].distribution
    assert csat_dist.kind == "categorical"
    assert csat_dist.params["categories"] == [1, 2, 3, 4, 5]
    assert all(isinstance(c, int) for c in csat_dist.params["categories"])


# --------------------------------------------------------------------------
# 2. Fit reproducibility: fitting the committed sample again from the
#    skeleton must reproduce the committed metadata.olist.yaml exactly
#    (within numeric tolerance).
# --------------------------------------------------------------------------


def test_fit_reproduces_committed_metadata(source_frames):
    skeleton = load_metadata(SKELETON_PATH)
    refit = fit_metadata(source_frames, skeleton)
    refit_dict = metadata_to_dict(refit)

    with open(FITTED_PATH) as f:
        committed_dict = yaml.safe_load(f)

    _assert_close(refit_dict, committed_dict)


# --------------------------------------------------------------------------
# 3. Generation + validation.
# --------------------------------------------------------------------------


def test_generate_and_validate(fitted_metadata, synth_out_dir):
    violations = validate_dataset(fitted_metadata, synth_out_dir)
    assert violations == []


def test_two_source_output_layout(synth_out_dir):
    """TASK CARD 12 §5.2: per-source output directories exist with parquet
    parts for every table."""
    for source, table in SOURCE_TABLES:
        table_dir = synth_out_dir / source / table
        assert table_dir.is_dir(), table_dir
        parts = sorted(table_dir.glob("part-*.parquet"))
        assert len(parts) == 2, f"{table_dir}: expected 2 partitions, found {len(parts)}"


# --------------------------------------------------------------------------
# 4. Statistical fidelity vs. the committed source sample.
# --------------------------------------------------------------------------


def test_customer_state_frequencies(source_frames, synth_frames):
    src_freq = _freq(source_frames["customers"]["customer_state"])
    syn_freq = _freq(synth_frames["customers"]["customer_state"])

    top3 = sorted(src_freq, key=src_freq.get, reverse=True)[:3]
    for state in top3:
        assert abs(src_freq[state] - syn_freq.get(state, 0.0)) < 0.03, state


def test_order_status_frequencies(source_frames, synth_frames):
    src_freq = _freq(source_frames["orders"]["order_status"])
    syn_freq = _freq(synth_frames["orders"]["order_status"])

    for status, p in src_freq.items():
        assert abs(p - syn_freq.get(status, 0.0)) < 0.03, status


def test_price_freight_copula_correlation(source_frames, synth_frames):
    src = source_frames["order_items"].select(["price", "freight_value"]).drop_nulls()
    syn = synth_frames["order_items"].select(["price", "freight_value"]).drop_nulls()

    src_rho, _ = stats.spearmanr(src["price"].to_numpy(), src["freight_value"].to_numpy())
    syn_rho, _ = stats.spearmanr(syn["price"].to_numpy(), syn["freight_value"].to_numpy())

    assert abs(src_rho - syn_rho) < 0.08


def test_installments_value_copula_correlation(source_frames, synth_frames):
    src = source_frames["order_payments"].select(["payment_installments", "payment_value"]).drop_nulls()
    syn = synth_frames["order_payments"].select(["payment_installments", "payment_value"]).drop_nulls()

    src_rho, _ = stats.spearmanr(
        src["payment_installments"].to_numpy(), src["payment_value"].to_numpy()
    )
    syn_rho, _ = stats.spearmanr(
        syn["payment_installments"].to_numpy(), syn["payment_value"].to_numpy()
    )

    assert abs(src_rho - syn_rho) < 0.08


def test_items_per_order_mean(source_frames, synth_frames):
    src_mean = source_frames["order_items"].height / source_frames["orders"].height
    syn_mean = synth_frames["order_items"].height / synth_frames["orders"].height

    assert abs(syn_mean - src_mean) / src_mean < 0.10


def test_review_score_distribution(source_frames, synth_frames):
    src_freq = _freq(source_frames["order_reviews"]["review_score"])
    syn_freq = _freq(synth_frames["order_reviews"]["review_score"])

    for score, p in src_freq.items():
        assert abs(p - syn_freq.get(score, 0.0)) < 0.03, score


def test_median_price_ratio(source_frames, synth_frames):
    src_median = source_frames["order_items"]["price"].median()
    syn_median = synth_frames["order_items"]["price"].median()

    ratio = syn_median / src_median
    assert 0.75 <= ratio <= 1.33


def test_median_approval_delay_ratio(source_frames, synth_frames):
    def _median_delay_seconds(df: pl.DataFrame) -> float:
        delta = (
            df.select(
                (
                    pl.col("order_approved_at") - pl.col("order_purchase_timestamp")
                ).alias("delta")
            )
            .drop_nulls()["delta"]
        )
        seconds = delta.dt.total_microseconds().to_numpy().astype(np.float64) / 1e6
        seconds = seconds[seconds >= 0]
        return float(np.median(seconds))

    src_median = _median_delay_seconds(source_frames["orders"])
    syn_median = _median_delay_seconds(synth_frames["orders"])

    ratio = syn_median / src_median
    assert 0.5 <= ratio <= 2.0


def test_temporal_ordering(synth_frames):
    orders = synth_frames["orders"]

    def _ge(later: str, earlier: str) -> None:
        sub = orders.select([later, earlier]).drop_nulls()
        assert (sub[later] >= sub[earlier]).all()

    _ge("order_delivered_customer_date", "order_delivered_carrier_date")
    _ge("order_delivered_carrier_date", "order_approved_at")
    _ge("order_approved_at", "order_purchase_timestamp")
    _ge("order_estimated_delivery_date", "order_purchase_timestamp")


# --------------------------------------------------------------------------
# 5. Fitted CRM fidelity vs. authored ground-truth constants
#    (TASK CARD 12 §5.1).
# --------------------------------------------------------------------------


def test_fitted_crm_categorical_fidelity(fitted_metadata, crm_prep_module):
    checks = [
        ("crm_contacts", "segment", crm_prep_module.SEGMENTS, crm_prep_module.SEGMENT_PROBS),
        ("crm_tickets", "channel", crm_prep_module.CHANNELS, crm_prep_module.CHANNEL_PROBS),
        ("crm_tickets", "category", crm_prep_module.CATEGORIES, crm_prep_module.CATEGORY_PROBS),
        ("crm_tickets", "priority", crm_prep_module.PRIORITIES, crm_prep_module.PRIORITY_PROBS),
        ("crm_tickets", "csat_score", crm_prep_module.CSAT_SCORES, crm_prep_module.CSAT_PROBS),
    ]
    for table, column, categories, probs in checks:
        fitted = _dist_probs(fitted_metadata, table, column)
        for cat, authored_p in zip(categories, probs):
            assert abs(fitted.get(cat, 0.0) - authored_p) < 0.02, (table, column, cat)


def test_fitted_bernoulli_p_matches_authored_leads_ratio(fitted_metadata):
    # 15,000 real shop customers / 25,000 total CRM contacts = 0.6 exactly.
    card = fitted_metadata.tables["customers"].cardinality
    assert card.kind == "bernoulli"
    assert abs(card.params["p"] - 0.6) < 0.005


def test_fitted_crm_state_matches_shop_master(fitted_metadata, source_frames):
    """Master fidelity: crm_contacts.state is fit from 15k real + leads
    drawn from the real empirical distribution, so it should track the real
    shop sample's customer_state distribution closely."""
    fitted = _dist_probs(fitted_metadata, "crm_contacts", "state")
    src_freq = _freq(source_frames["customers"]["customer_state"])
    for state, p in src_freq.items():
        assert abs(fitted.get(state, 0.0) - p) < 0.02, state


# --------------------------------------------------------------------------
# 6. Master-data consistency: the core cross-source assertion
#    (TASK CARD 12 §5.3).
# --------------------------------------------------------------------------


def test_master_data_consistency(fitted_metadata, synth_out_dir):
    backbone = ParquetBackbone(synth_out_dir)
    cust_glob = backbone.table_glob("customers", fitted_metadata.tables["customers"].source)
    contacts_glob = backbone.table_glob("crm_contacts", fitted_metadata.tables["crm_contacts"].source)
    orders_glob = backbone.table_glob("orders", fitted_metadata.tables["orders"].source)

    con = duckdb.connect()
    try:
        # customer_state must equal the owning CRM contact's state for
        # EVERY shop customer: zero exceptions, by construction
        # (generator: parent:state), not statistically.
        (mismatches,) = con.execute(
            f"""
            SELECT count(*) FROM read_parquet('{cust_glob}') c
            JOIN read_parquet('{contacts_glob}') p ON c.contact_id = p.contact_id
            WHERE c.customer_state != p.state
            """
        ).fetchone()
        assert mismatches == 0

        # Every orders.customer_id exists among shop customers (already
        # covered by validate_dataset -- kept explicit here too).
        (orphans,) = con.execute(
            f"""
            SELECT count(*) FROM read_parquet('{orders_glob}') o
            LEFT JOIN read_parquet('{cust_glob}') c ON o.customer_id = c.customer_id
            WHERE c.customer_id IS NULL
            """
        ).fetchone()
        assert orphans == 0

        (n_contacts,) = con.execute(f"SELECT count(*) FROM read_parquet('{contacts_glob}')").fetchone()
        (n_customers,) = con.execute(f"SELECT count(*) FROM read_parquet('{cust_glob}')").fetchone()
    finally:
        con.close()

    p_fitted = fitted_metadata.tables["customers"].cardinality.params["p"]
    assert abs(n_customers / n_contacts - p_fitted) < 0.02


# --------------------------------------------------------------------------
# 7. CRM temporal ordering + null structure (TASK CARD 12 §5.4).
# --------------------------------------------------------------------------


def test_crm_temporal_ordering_and_null_structure(synth_frames, crm_prep_module):
    contacts = synth_frames["crm_contacts"]
    tickets = synth_frames["crm_tickets"]

    joined = tickets.join(
        contacts.select(["contact_id", pl.col("created_at").alias("contact_created_at")]),
        on="contact_id",
        how="left",
    )
    assert (joined["opened_at"] >= joined["contact_created_at"]).all()

    resolved_valid = tickets.select(["opened_at", "resolved_at"]).drop_nulls()
    assert (resolved_valid["resolved_at"] >= resolved_valid["opened_at"]).all()

    resolved_null_frac = tickets["resolved_at"].is_null().mean()
    assert abs(resolved_null_frac - crm_prep_module.RESOLVED_NULL_RATE) < 0.02

    csat_null_frac = tickets["csat_score"].is_null().mean()
    assert abs(csat_null_frac - crm_prep_module.CSAT_NULL_RATE) < 0.02

    csat_freq = _freq(tickets["csat_score"].drop_nulls())
    for score, p in zip(crm_prep_module.CSAT_SCORES, crm_prep_module.CSAT_PROBS):
        assert abs(csat_freq.get(score, 0.0) - p) < 0.03


# --------------------------------------------------------------------------
# 8. Determinism (TASK CARD 12 §5.5).
# --------------------------------------------------------------------------


def test_determinism(fitted_metadata):
    engine1 = Engine(load_metadata(FITTED_PATH))
    engine2 = Engine(load_metadata(FITTED_PATH))

    tables1 = engine1.generate_partition(0, 2)
    tables2 = engine2.generate_partition(0, 2)

    assert set(tables1) == set(tables2)
    for tname in tables1:
        assert tables1[tname].equals(tables2[tname]), tname


def test_determinism_across_sources_and_partitions(fitted_metadata):
    engine1 = Engine(load_metadata(FITTED_PATH))
    engine2 = Engine(load_metadata(FITTED_PATH))
    single = engine1.generate_partition(0, 1)
    single2 = engine2.generate_partition(0, 1)

    assert single["crm_contacts"].equals(single2["crm_contacts"])
    assert single["customers"].equals(single2["customers"])

    # Inherited-state guarantee holds per-partition: P=1 equals the
    # concatenation of P=3 (docs/ARCHITECTURE.md §3).
    engine3 = Engine(load_metadata(FITTED_PATH))
    parts = [engine3.generate_partition(p, 3) for p in range(3)]
    concatenated = pa.concat_tables([parts[p]["customers"] for p in range(3)])
    assert concatenated.equals(single["customers"])


# --------------------------------------------------------------------------
# 9. Third source: inventory (master product catalog + shipment orders,
#    TASK CARD 14, docs/ARCHITECTURE.md §2, §8).
# --------------------------------------------------------------------------


def test_inv_products_row_count_matches_distinct_products(fitted_metadata, source_frames):
    skeleton = load_metadata(SKELETON_PATH)
    distinct_products = source_frames["order_items"]["product_id"].n_unique()

    assert skeleton.tables["inv_products"].rows == distinct_products
    assert source_frames["inv_products"].height == distinct_products
    assert fitted_metadata.tables["inv_products"].rows == distinct_products


def test_fitted_shipments_bernoulli_p_matches_observed_share(fitted_metadata, source_frames):
    orders = source_frames["orders"]
    observed_share = orders["order_status"].is_in(["delivered", "shipped"]).mean()

    card = fitted_metadata.tables["inv_shipments"].cardinality
    assert card.kind == "bernoulli"
    print(f"fitted bernoulli p = {card.params['p']!r}, observed share = {observed_share!r}")
    assert abs(card.params["p"] - observed_share) < 0.01


def test_fitted_inventory_categorical_fidelity(fitted_metadata, inventory_prep_module):
    checks = [
        ("inv_shipments", "warehouse", inventory_prep_module.WAREHOUSES, inventory_prep_module.WAREHOUSE_PROBS),
        ("inv_shipments", "carrier", inventory_prep_module.CARRIERS, inventory_prep_module.CARRIER_PROBS),
    ]
    for table, column, categories, probs in checks:
        fitted = _dist_probs(fitted_metadata, table, column)
        for cat, authored_p in zip(categories, probs):
            assert abs(fitted.get(cat, 0.0) - authored_p) < 0.02, (table, column, cat)


def test_fitted_inv_products_category_matches_real_frequencies(fitted_metadata, source_frames):
    fitted = _dist_probs(fitted_metadata, "inv_products", "category")
    src_freq = _freq(source_frames["inv_products"]["category"])
    for category, p in src_freq.items():
        assert abs(fitted.get(category, 0.0) - p) < 0.02, category


def test_fitted_product_id_zipf_a_in_range(fitted_metadata):
    # Finite zipfian is well-defined for a >= 0 (a = 0 is uniform over
    # ranks, docs/ARCHITECTURE.md §2); the real Olist per-product
    # popularity in this sample is close to flat, so the fitted a should
    # land well below 1, not saturate at some a > 1 grid floor.
    dist = fitted_metadata.tables["order_items"].columns["product_id"].distribution
    assert dist.kind == "zipf"
    a = dist.params["a"]
    print(f"fitted zipf a = {a!r}")
    assert 0.0 <= a <= 3.5
    assert dist.params["n"] == fitted_metadata.tables["inv_products"].rows


def test_fitted_shipment_delays_match_authored(fitted_metadata, inventory_prep_module):
    checks = [
        ("created_at", inventory_prep_module.CREATED_AT_DELAY_MU, inventory_prep_module.CREATED_AT_DELAY_SIGMA),
        ("picked_at", inventory_prep_module.PICKED_AT_DELAY_MU, inventory_prep_module.PICKED_AT_DELAY_SIGMA),
        ("handed_over_at", inventory_prep_module.HANDED_OVER_DELAY_MU, inventory_prep_module.HANDED_OVER_DELAY_SIGMA),
    ]
    for column, mu, sigma in checks:
        delay = fitted_metadata.tables["inv_shipments"].columns[column].temporal.delay
        assert delay.kind == "lognormal"
        assert abs(delay.params["mu"] - mu) < 0.05, column
        assert abs(delay.params["sigma"] - sigma) < 0.05, column


def test_inventory_output_layout(synth_out_dir):
    for table in ("inv_products", "inv_shipments"):
        table_dir = synth_out_dir / "inventory" / table
        assert table_dir.is_dir(), table_dir
        parts = sorted(table_dir.glob("part-*.parquet"))
        assert len(parts) == 2, f"{table_dir}: expected 2 partitions, found {len(parts)}"


def test_shipments_order_leading_and_strictly_later(fitted_metadata, synth_out_dir):
    """Order-leading / strictly-later semantics (docs/ARCHITECTURE.md §8):
    every inv_shipments row references a real shop order, and its
    created_at is strictly after the order's purchase timestamp -- never
    equal, never before."""
    backbone = ParquetBackbone(synth_out_dir)
    shipments_glob = backbone.table_glob("inv_shipments", fitted_metadata.tables["inv_shipments"].source)
    orders_glob = backbone.table_glob("orders", fitted_metadata.tables["orders"].source)

    con = duckdb.connect()
    try:
        (n_shipments,) = con.execute(f"SELECT count(*) FROM read_parquet('{shipments_glob}')").fetchone()
        (n_matched,) = con.execute(
            f"""
            SELECT count(*) FROM read_parquet('{shipments_glob}') s
            JOIN read_parquet('{orders_glob}') o ON s.order_id = o.order_id
            """
        ).fetchone()
        assert n_matched == n_shipments

        (violations,) = con.execute(
            f"""
            SELECT count(*) FROM read_parquet('{shipments_glob}') s
            JOIN read_parquet('{orders_glob}') o ON s.order_id = o.order_id
            WHERE s.created_at <= o.order_purchase_timestamp
            """
        ).fetchone()
        print(f"strictly-later violations = {violations}")
        assert violations == 0

        (order_violations,) = con.execute(
            f"""
            SELECT count(*) FROM read_parquet('{shipments_glob}')
            WHERE NOT (handed_over_at >= picked_at AND picked_at >= created_at)
            """
        ).fetchone()
        assert order_violations == 0
    finally:
        con.close()


def test_master_feeds_shop_product_reference(fitted_metadata, synth_out_dir, synth_frames, source_frames):
    """Master fidelity for the dimension-reference pattern (docs/ARCHITECTURE.md
    §2, §8): every synthetic order_items.product_id resolves to a real
    generated inv_products row, and the synthetic popularity of the
    top-ranked product tracks the *real* committed sample's top-product
    share (not merely the fitted pmf it was derived from -- a genuine
    real-data-fidelity check)."""
    backbone = ParquetBackbone(synth_out_dir)
    items_glob = backbone.table_glob("order_items", fitted_metadata.tables["order_items"].source)
    products_glob = backbone.table_glob("inv_products", fitted_metadata.tables["inv_products"].source)

    n_products = fitted_metadata.tables["inv_products"].rows

    product_ref = synth_frames["order_items"]["product_id"].to_numpy()
    assert product_ref.min() >= 0
    assert product_ref.max() < n_products

    con = duckdb.connect()
    try:
        (missing,) = con.execute(
            f"""
            SELECT count(*) FROM read_parquet('{items_glob}') i
            LEFT JOIN read_parquet('{products_glob}') p ON i.product_id = p.product_id
            WHERE p.product_id IS NULL
            """
        ).fetchone()
        assert missing == 0
    finally:
        con.close()

    src_counts = source_frames["order_items"]["product_id"].value_counts()
    src_top_share = float(src_counts["count"].max()) / source_frames["order_items"].height

    rank0_share = float((synth_frames["order_items"]["product_id"] == 0).mean())
    print(
        f"synthetic rank-0 empirical share = {rank0_share!r}, "
        f"committed-sample real top-product share = {src_top_share!r}"
    )
    assert abs(rank0_share - src_top_share) < 0.02

    n_shipments = synth_frames["inv_shipments"].height
    n_orders = synth_frames["orders"].height
    p_fitted = fitted_metadata.tables["inv_shipments"].cardinality.params["p"]
    assert abs(n_shipments / n_orders - p_fitted) < 0.02


def test_determinism_inv_shipments(fitted_metadata):
    """Cross-source determinism (TASK CARD 14 §6): two independent Engines
    over the same fitted metadata produce byte-identical inv_shipments."""
    engine1 = Engine(load_metadata(FITTED_PATH))
    engine2 = Engine(load_metadata(FITTED_PATH))

    tables1 = engine1.generate_partition(0, 2)
    tables2 = engine2.generate_partition(0, 2)

    assert tables1["inv_shipments"].equals(tables2["inv_shipments"])
    assert tables1["inv_products"].equals(tables2["inv_products"])
