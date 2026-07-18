"""Acceptance tests for verisynth.fit (fitted-parameter release, optional DP).

See docs/ARCHITECTURE.md §7 (normative) and TASK CARD 7.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from verisynth.fit import fit_metadata
from verisynth.metadata import load_metadata, metadata_to_dict, parse_metadata

REPO_ROOT = Path(__file__).resolve().parent.parent
RETAIL_YAML = REPO_ROOT / "examples" / "retail.yaml"

N_CUSTOMERS = 20_000

TRUE_AGE_MEAN, TRUE_AGE_STD = 41.0, 13.0
TRUE_INCOME_MU, TRUE_INCOME_SIGMA = 10.8, 0.6
TRUE_COPULA_R = 0.55
TRUE_REGION_PROBS = {"NA": 0.5, "EU": 0.3, "APAC": 0.2}
TRUE_CARD_LAM = 4.2
TRUE_ORDER_TOTAL_MU, TRUE_ORDER_TOTAL_SIGMA = 3.4, 0.8
TRUE_ORDERED_DELAY_MEAN_S = 3 * 86400.0  # exponential mean delay: 3 days
TRUE_SHIPPED_MU, TRUE_SHIPPED_SIGMA = 11.5, 0.6


def _generate_frames(seed: int = 0, n_customers: int = N_CUSTOMERS):
    """Generate "real" customers/orders data matching examples/retail.yaml's
    structure, with known ground-truth generating parameters."""
    rng = np.random.default_rng(seed)

    # Correlated latent normals (Pearson r = TRUE_COPULA_R) -> Gaussian copula
    # construction for (age, income).
    cov = [[1.0, TRUE_COPULA_R], [TRUE_COPULA_R, 1.0]]
    z = rng.multivariate_normal([0.0, 0.0], cov, size=n_customers)
    age = TRUE_AGE_MEAN + TRUE_AGE_STD * z[:, 0]
    income = np.exp(TRUE_INCOME_MU + TRUE_INCOME_SIGMA * z[:, 1])

    customer_id = np.arange(n_customers, dtype=np.int64)
    region = rng.choice(
        list(TRUE_REGION_PROBS.keys()), size=n_customers, p=list(TRUE_REGION_PROBS.values())
    )

    signup_start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    signup_end = datetime(2024, 1, 1, tzinfo=timezone.utc)
    signup_range_s = (signup_end - signup_start).total_seconds()
    signup_offset_s = rng.uniform(0.0, signup_range_s, size=n_customers)
    signup_us = (
        int(signup_start.timestamp()) * 1_000_000 + (signup_offset_s * 1e6).astype(np.int64)
    )

    customers_df = pl.DataFrame(
        {
            "customer_id": customer_id,
            "region": region,
            "age": age,
            "income": income,
            "signup_at": signup_us.astype("datetime64[us]"),
        }
    )

    counts = rng.poisson(TRUE_CARD_LAM, size=n_customers).astype(np.int64)
    order_customer_idx = np.repeat(np.arange(n_customers), counts)
    n_orders = order_customer_idx.size
    order_id = np.arange(n_orders, dtype=np.int64)
    order_customer_id = customer_id[order_customer_idx]
    signup_us_per_order = signup_us[order_customer_idx]

    delay1_s = rng.exponential(scale=TRUE_ORDERED_DELAY_MEAN_S, size=n_orders)
    ordered_us = signup_us_per_order + (delay1_s * 1e6).astype(np.int64)

    z2 = rng.normal(size=n_orders)
    delay2_s = np.exp(TRUE_SHIPPED_MU + TRUE_SHIPPED_SIGMA * z2)
    shipped_us = ordered_us + (delay2_s * 1e6).astype(np.int64)

    order_total = np.exp(rng.normal(TRUE_ORDER_TOTAL_MU, TRUE_ORDER_TOTAL_SIGMA, size=n_orders))

    orders_df = pl.DataFrame(
        {
            "order_id": order_id,
            "customer_id": order_customer_id,
            "order_total": order_total,
            "ordered_at": ordered_us.astype("datetime64[us]"),
            "shipped_at": shipped_us.astype("datetime64[us]"),
        }
    )

    extras = {"delay1_s": delay1_s, "delay2_s": delay2_s, "counts": counts}
    return {"customers": customers_df, "orders": orders_df}, extras


@pytest.fixture(scope="module")
def frames_and_extras():
    return _generate_frames()


# --------------------------------------------------------------------------
# 1. No-DP fit recovers the generating parameters
# --------------------------------------------------------------------------


def test_fit_no_dp_recovers_generating_params(frames_and_extras):
    frames, extras = frames_and_extras
    skeleton = load_metadata(str(RETAIL_YAML))

    fitted = fit_metadata(frames, skeleton)

    age_dist = fitted.tables["customers"].columns["age"].distribution
    assert age_dist.kind == "normal"
    assert abs(age_dist.params["mean"] - TRUE_AGE_MEAN) < 1.0
    assert abs(age_dist.params["std"] - TRUE_AGE_STD) < 0.5

    income_dist = fitted.tables["customers"].columns["income"].distribution
    assert income_dist.kind == "lognormal"
    assert abs(income_dist.params["mu"] - TRUE_INCOME_MU) < 0.05
    assert abs(income_dist.params["sigma"] - TRUE_INCOME_SIGMA) < 0.05

    region_dist = fitted.tables["customers"].columns["region"].distribution
    assert region_dist.kind == "categorical"
    fitted_probs = dict(zip(region_dist.params["categories"], region_dist.params["probs"]))
    for cat, p in TRUE_REGION_PROBS.items():
        assert abs(fitted_probs[cat] - p) < 0.02

    card = fitted.tables["orders"].cardinality
    assert card.kind == "poisson"
    assert abs(card.params["lam"] - TRUE_CARD_LAM) < 0.1

    copula = fitted.tables["customers"].copulas[0]
    assert copula.columns == ["age", "income"]
    assert abs(copula.correlation[0][1] - TRUE_COPULA_R) < 0.05
    assert abs(copula.correlation[1][0] - TRUE_COPULA_R) < 0.05
    assert copula.correlation[0][0] == pytest.approx(1.0)
    assert copula.correlation[1][1] == pytest.approx(1.0)

    # ordered_at delay: assert whichever kind the fit rule selects on this
    # generated data (>= 95% of observed delays > 0 -> robust lognormal fitted
    # on the strictly-positive subset with mu = median(log d), sigma =
    # std(log d, ddof=0); else exponential over all delays), and check the
    # params against the same statistic computed independently here.
    d1 = extras["delay1_s"]
    ordered_delay = fitted.tables["orders"].columns["ordered_at"].temporal.delay
    if np.mean(d1 > 0) >= 0.95:
        assert ordered_delay.kind == "lognormal"
        d1_pos = d1[d1 > 0]
        expected_mu = float(np.median(np.log(d1_pos)))
        expected_sigma = float(np.std(np.log(d1_pos), ddof=0))
        assert abs(ordered_delay.params["mu"] - expected_mu) < 0.05
        assert abs(ordered_delay.params["sigma"] - expected_sigma) < 0.05
    else:
        assert ordered_delay.kind == "exponential"
        expected_rate = 1.0 / max(float(np.mean(d1)), 1e-9)
        assert abs(ordered_delay.params["rate"] - expected_rate) / expected_rate < 0.1

    # shipped_at delay: lognormal-generated delay is (for all practical
    # purposes) always strictly positive, so the rule always selects lognormal.
    # For genuinely lognormal-generated data median(log) ~= mean(log) = mu_true,
    # so the mu tolerance below still holds under the robust (median-of-logs) fit.
    d2 = extras["delay2_s"]
    shipped_delay = fitted.tables["orders"].columns["shipped_at"].temporal.delay
    assert shipped_delay.kind == "lognormal"
    assert abs(shipped_delay.params["mu"] - TRUE_SHIPPED_MU) < 0.05
    expected_shipped_sigma = float(np.std(np.log(d2), ddof=0))
    assert abs(shipped_delay.params["sigma"] - expected_shipped_sigma) < 0.05


# --------------------------------------------------------------------------
# 2. Round-trip through metadata_to_dict / parse_metadata / YAML
# --------------------------------------------------------------------------


def test_fit_round_trip(frames_and_extras, tmp_path):
    frames, _ = frames_and_extras
    skeleton = load_metadata(str(RETAIL_YAML))

    fitted = fit_metadata(frames, skeleton)

    d = metadata_to_dict(fitted)
    reparsed = parse_metadata(d)
    assert metadata_to_dict(reparsed) == d

    out_path = tmp_path / "fitted.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(d, f)

    reloaded = load_metadata(str(out_path))
    assert metadata_to_dict(reloaded) == d


# --------------------------------------------------------------------------
# 3. Determinism
# --------------------------------------------------------------------------


def test_fit_determinism(frames_and_extras):
    frames, _ = frames_and_extras
    skeleton = load_metadata(str(RETAIL_YAML))

    fitted1 = fit_metadata(frames, skeleton)
    fitted2 = fit_metadata(frames, skeleton)
    assert metadata_to_dict(fitted1) == metadata_to_dict(fitted2)

    skeleton_dp = copy.deepcopy(skeleton)
    skeleton_dp.tables["customers"].columns["income"].clamp = (1.0, 1e7)
    skeleton_dp.tables["orders"].columns["order_total"].clamp = (0.01, 1e6)

    dp1 = fit_metadata(frames, skeleton_dp, epsilon=0.5, dp_seed=7)
    dp2 = fit_metadata(frames, skeleton_dp, epsilon=0.5, dp_seed=7)
    assert metadata_to_dict(dp1) == metadata_to_dict(dp2)

    dp3 = fit_metadata(frames, skeleton_dp, epsilon=0.5, dp_seed=8)
    assert metadata_to_dict(dp3) != metadata_to_dict(dp2)


# --------------------------------------------------------------------------
# 4. DP behavior
# --------------------------------------------------------------------------


def test_fit_dp_behavior(frames_and_extras):
    frames, _ = frames_and_extras
    skeleton = load_metadata(str(RETAIL_YAML))

    no_dp = fit_metadata(frames, skeleton)

    skeleton_dp = copy.deepcopy(skeleton)
    skeleton_dp.tables["customers"].columns["income"].clamp = (1.0, 1e7)
    skeleton_dp.tables["orders"].columns["order_total"].clamp = (0.01, 1e6)

    dp = fit_metadata(frames, skeleton_dp, epsilon=0.5, dp_seed=0)

    assert metadata_to_dict(dp) != metadata_to_dict(no_dp)

    region_dist = dp.tables["customers"].columns["region"].distribution
    probs = region_dist.params["probs"]
    assert all(p >= 0 for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-6

    copula = dp.tables["customers"].copulas[0]
    corr = copula.correlation
    assert corr[0][0] == pytest.approx(1.0)
    assert corr[1][1] == pytest.approx(1.0)
    assert corr[0][1] == pytest.approx(corr[1][0])
    assert -0.99 <= corr[0][1] <= 0.99

    # DP-fitted metadata must still validate and round-trip.
    parse_metadata(metadata_to_dict(dp))

    # Missing clamp on a numeric column under DP -> ValueError naming the column
    # (plain `skeleton`: income has no clamp).
    with pytest.raises(ValueError, match="income"):
        fit_metadata(frames, skeleton, epsilon=0.5, dp_seed=0)


# --------------------------------------------------------------------------
# 5. Never mutates the skeleton
# --------------------------------------------------------------------------


def test_fit_never_mutates_skeleton(frames_and_extras):
    frames, _ = frames_and_extras
    skeleton = load_metadata(str(RETAIL_YAML))
    skeleton_snapshot = copy.deepcopy(skeleton)

    fit_metadata(frames, skeleton)
    assert skeleton == skeleton_snapshot

    skeleton_dp = copy.deepcopy(skeleton)
    skeleton_dp.tables["customers"].columns["income"].clamp = (1.0, 1e7)
    skeleton_dp.tables["orders"].columns["order_total"].clamp = (0.01, 1e6)
    dp_snapshot = copy.deepcopy(skeleton_dp)

    fit_metadata(frames, skeleton_dp, epsilon=0.5, dp_seed=3)
    assert skeleton_dp == dp_snapshot


# --------------------------------------------------------------------------
# 6. Declared-categorical int64 columns fit as categorical (TASK CARD 9 §1)
# --------------------------------------------------------------------------


def test_fit_int_dtype_column_declared_categorical_stays_categorical():
    skeleton = parse_metadata(
        {
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
                                "categories": [1, 2, 3, 4, 5],
                                "probs": [0.2, 0.2, 0.2, 0.2, 0.2],
                            },
                        },
                    },
                }
            },
        }
    )
    df = pl.DataFrame({"id": [0, 1, 2, 3, 4], "score": pl.Series([1, 1, 2, 3, 5], dtype=pl.Int64)})

    fitted = fit_metadata({"t": df}, skeleton)

    dist = fitted.tables["t"].columns["score"].distribution
    assert dist.kind == "categorical"
    assert dist.params["categories"] == [1, 2, 3, 5]
    assert all(isinstance(c, int) for c in dist.params["categories"])

    expected = {1: 0.4, 2: 0.2, 3: 0.2, 5: 0.2}
    for cat, p in zip(dist.params["categories"], dist.params["probs"]):
        assert abs(p - expected[cat]) < 1e-9


# --------------------------------------------------------------------------
# 7. Revised temporal-delay fit rule (TASK CARD 10 / docs/ARCHITECTURE.md §7):
#    >= 95% of observed delays > 0 -> robust lognormal fitted on the
#    strictly-positive subset (mu = median(log d), sigma = std(log d, ddof=0)
#    -- median-preserving by construction), else exponential with
#    rate = 1/mean over all delays.
# --------------------------------------------------------------------------


def _build_customer_order_frames(delay_s: np.ndarray, seed: int = 1) -> dict[str, pl.DataFrame]:
    """Build minimal customers/orders frames (matching examples/retail.yaml's
    structure) whose ordered_at = signup_at + delay_s exactly, so the
    temporal-delay fit rule can be exercised on a caller-controlled delay
    array (including exact zeros)."""
    rng = np.random.default_rng(seed)
    n_orders = delay_s.size

    signup_start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    signup_us = np.full(n_orders, int(signup_start.timestamp()) * 1_000_000, dtype=np.int64)

    customer_id = np.arange(n_orders, dtype=np.int64)
    region = rng.choice(["NA", "EU", "APAC"], size=n_orders, p=[0.5, 0.3, 0.2])
    age = rng.normal(TRUE_AGE_MEAN, TRUE_AGE_STD, size=n_orders)
    income = np.exp(rng.normal(TRUE_INCOME_MU, TRUE_INCOME_SIGMA, size=n_orders))

    customers_df = pl.DataFrame(
        {
            "customer_id": customer_id,
            "region": region,
            "age": age,
            "income": income,
            "signup_at": signup_us.astype("datetime64[us]"),
        }
    )

    order_id = np.arange(n_orders, dtype=np.int64)
    ordered_us = signup_us + (delay_s * 1e6).astype(np.int64)
    z2 = rng.normal(size=n_orders)
    delay2_s = np.exp(TRUE_SHIPPED_MU + TRUE_SHIPPED_SIGMA * z2)
    shipped_us = ordered_us + (delay2_s * 1e6).astype(np.int64)
    order_total = np.exp(rng.normal(TRUE_ORDER_TOTAL_MU, TRUE_ORDER_TOTAL_SIGMA, size=n_orders))

    orders_df = pl.DataFrame(
        {
            "order_id": order_id,
            "customer_id": customer_id,
            "order_total": order_total,
            "ordered_at": ordered_us.astype("datetime64[us]"),
            "shipped_at": shipped_us.astype("datetime64[us]"),
        }
    )
    return {"customers": customers_df, "orders": orders_df}


def test_fit_temporal_delay_small_zero_fraction_stays_lognormal():
    # 2% exact-zero delays, 98% strictly-positive lognormal delays: below the
    # 95% threshold's complement, so the rule must still fit lognormal on the
    # strictly-positive subset (robust to a small fraction of zero delays).
    rng = np.random.default_rng(2)
    n = 20_000
    log_mu, log_sigma = 11.0, 0.5
    delay_s = np.exp(rng.normal(log_mu, log_sigma, size=n))
    n_zero = int(round(0.02 * n))
    zero_idx = rng.choice(n, size=n_zero, replace=False)
    delay_s[zero_idx] = 0.0

    frames = _build_customer_order_frames(delay_s, seed=3)
    skeleton = load_metadata(str(RETAIL_YAML))

    fitted = fit_metadata(frames, skeleton)
    ordered_delay = fitted.tables["orders"].columns["ordered_at"].temporal.delay

    assert ordered_delay.kind == "lognormal"
    d_pos = delay_s[delay_s > 0]
    expected_mu = float(np.median(np.log(d_pos)))
    assert abs(ordered_delay.params["mu"] - expected_mu) < 0.05


def test_fit_temporal_delay_large_zero_fraction_selects_exponential():
    # 20% exact-zero delays: well above the 5% zero-tolerance, so the rule
    # must fall back to exponential fit with rate = 1/mean over all delays.
    rng = np.random.default_rng(4)
    n = 20_000
    log_mu, log_sigma = 11.0, 0.5
    delay_s = np.exp(rng.normal(log_mu, log_sigma, size=n))
    n_zero = int(round(0.20 * n))
    zero_idx = rng.choice(n, size=n_zero, replace=False)
    delay_s[zero_idx] = 0.0

    frames = _build_customer_order_frames(delay_s, seed=5)
    skeleton = load_metadata(str(RETAIL_YAML))

    fitted = fit_metadata(frames, skeleton)
    ordered_delay = fitted.tables["orders"].columns["ordered_at"].temporal.delay

    assert ordered_delay.kind == "exponential"
    expected_rate = 1.0 / max(float(np.mean(delay_s)), 1e-9)
    assert abs(ordered_delay.params["rate"] - expected_rate) / expected_rate < 0.05


# --------------------------------------------------------------------------
# 8. bernoulli cardinality fit rule (TASK CARD 11 §B / docs/ARCHITECTURE.md §7):
#    if every observed per-parent count is 0 or 1 -> bernoulli{p = mean},
#    child_stride = 2. A mixed-count frame still fits poisson (existing
#    tests above already cover that).
# --------------------------------------------------------------------------


def _generate_bernoulli_frames(p: float, seed: int = 0, n_customers: int = N_CUSTOMERS):
    """Like `_generate_frames`, but each customer has 0 or 1 orders with
    probability `p` (a "CRM contact has a shop account" style relationship)."""
    rng = np.random.default_rng(seed)

    cov = [[1.0, TRUE_COPULA_R], [TRUE_COPULA_R, 1.0]]
    z = rng.multivariate_normal([0.0, 0.0], cov, size=n_customers)
    age = TRUE_AGE_MEAN + TRUE_AGE_STD * z[:, 0]
    income = np.exp(TRUE_INCOME_MU + TRUE_INCOME_SIGMA * z[:, 1])

    customer_id = np.arange(n_customers, dtype=np.int64)
    region = rng.choice(
        list(TRUE_REGION_PROBS.keys()), size=n_customers, p=list(TRUE_REGION_PROBS.values())
    )

    signup_start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    signup_end = datetime(2024, 1, 1, tzinfo=timezone.utc)
    signup_range_s = (signup_end - signup_start).total_seconds()
    signup_offset_s = rng.uniform(0.0, signup_range_s, size=n_customers)
    signup_us = (
        int(signup_start.timestamp()) * 1_000_000 + (signup_offset_s * 1e6).astype(np.int64)
    )

    customers_df = pl.DataFrame(
        {
            "customer_id": customer_id,
            "region": region,
            "age": age,
            "income": income,
            "signup_at": signup_us.astype("datetime64[us]"),
        }
    )

    counts = rng.binomial(1, p, size=n_customers).astype(np.int64)
    order_customer_idx = np.repeat(np.arange(n_customers), counts)
    n_orders = order_customer_idx.size
    order_id = np.arange(n_orders, dtype=np.int64)
    order_customer_id = customer_id[order_customer_idx]
    signup_us_per_order = signup_us[order_customer_idx]

    delay1_s = rng.exponential(scale=TRUE_ORDERED_DELAY_MEAN_S, size=n_orders)
    ordered_us = signup_us_per_order + (delay1_s * 1e6).astype(np.int64)

    z2 = rng.normal(size=n_orders)
    delay2_s = np.exp(TRUE_SHIPPED_MU + TRUE_SHIPPED_SIGMA * z2)
    shipped_us = ordered_us + (delay2_s * 1e6).astype(np.int64)

    order_total = np.exp(rng.normal(TRUE_ORDER_TOTAL_MU, TRUE_ORDER_TOTAL_SIGMA, size=n_orders))

    orders_df = pl.DataFrame(
        {
            "order_id": order_id,
            "customer_id": order_customer_id,
            "order_total": order_total,
            "ordered_at": ordered_us.astype("datetime64[us]"),
            "shipped_at": shipped_us.astype("datetime64[us]"),
        }
    )

    return {"customers": customers_df, "orders": orders_df}, counts


def test_fit_cardinality_selects_bernoulli_when_counts_are_0_or_1():
    p_true = 0.6
    frames, counts = _generate_bernoulli_frames(p_true, seed=11)
    skeleton = load_metadata(str(RETAIL_YAML))

    fitted = fit_metadata(frames, skeleton)

    card = fitted.tables["orders"].cardinality
    assert card.kind == "bernoulli"
    assert abs(card.params["p"] - p_true) < 0.02
    assert fitted.tables["orders"].child_stride == 2

    # sanity: the frame really is 0/1-only.
    assert set(np.unique(counts)) <= {0, 1}

    # the fitted metadata must still round-trip/validate.
    d = metadata_to_dict(fitted)
    reparsed = parse_metadata(d)
    assert metadata_to_dict(reparsed) == d


def test_fit_cardinality_mixed_counts_still_fits_poisson(frames_and_extras):
    # frames_and_extras uses TRUE_CARD_LAM = 4.2 (poisson), so most parents
    # have counts > 1 -> mixed-count frame, must NOT be classified bernoulli.
    frames, _ = frames_and_extras
    skeleton = load_metadata(str(RETAIL_YAML))

    fitted = fit_metadata(frames, skeleton)
    card = fitted.tables["orders"].cardinality
    assert card.kind == "poisson"
    assert abs(card.params["lam"] - TRUE_CARD_LAM) < 0.1
