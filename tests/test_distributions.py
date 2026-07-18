"""Acceptance tests for verisynth.distributions marginal samplers."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from verisynth.kernels import keyed_uniforms
from verisynth.distributions import make_marginal, make_delay_ppf
from verisynth.metadata import DistributionSpec, CardinalitySpec


N = 20000


def _u(seed=1, ns="t.c"):
    return keyed_uniforms(seed, ns, np.arange(N, dtype=np.uint64))


def test_normal_moments():
    spec = DistributionSpec(kind="normal", params={"mean": 41.0, "std": 13.0})
    vals = make_marginal(spec).ppf(_u())
    assert abs(np.mean(vals) - 41.0) < 0.02 * 41.0
    assert abs(np.std(vals) - 13.0) < 0.02 * 13.0


def test_lognormal_median():
    spec = DistributionSpec(kind="lognormal", params={"mu": 10.8, "sigma": 0.6})
    vals = make_marginal(spec).ppf(_u())
    median = np.median(vals)
    expected = np.exp(10.8)
    assert abs(median - expected) < 0.03 * expected


def test_exponential_mean():
    rate = 1.0e-3
    spec = DistributionSpec(kind="exponential", params={"rate": rate})
    vals = make_marginal(spec).ppf(_u())
    expected = 1.0 / rate
    assert abs(np.mean(vals) - expected) < 0.03 * expected
    assert np.all(vals >= 0)


def test_uniform_bounds():
    spec = DistributionSpec(kind="uniform", params={"low": 5.0, "high": 15.0})
    vals = make_marginal(spec).ppf(_u())
    assert np.all(vals >= 5.0) and np.all(vals <= 15.0)
    assert abs(np.mean(vals) - 10.0) < 0.02 * 10.0


def test_categorical_frequencies():
    categories = ["NA", "EU", "APAC"]
    probs = [0.5, 0.3, 0.2]
    spec = DistributionSpec(kind="categorical", params={"categories": categories, "probs": probs})
    vals = make_marginal(spec).ppf(_u())
    for cat, p in zip(categories, probs):
        freq = np.mean(vals == cat)
        assert abs(freq - p) < 0.01


def test_uniform_int_covers_range_and_bounds():
    spec = DistributionSpec(kind="uniform_int", params={"low": 1, "high": 6})
    vals = make_marginal(spec).ppf(_u())
    assert vals.dtype == np.int64
    assert set(np.unique(vals)) == {1, 2, 3, 4, 5, 6}
    assert np.all(vals >= 1) and np.all(vals <= 6)


def test_datetime_uniform_range():
    start = "2022-01-01T00:00:00"
    end = "2025-12-31T23:59:59"
    spec = DistributionSpec(kind="datetime_uniform", params={"start": start, "end": end})
    vals = make_marginal(spec).ppf(_u())
    start_us = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp() * 1e6)
    end_us = int(datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp() * 1e6)
    assert vals.dtype == np.int64
    assert np.all(vals >= start_us) and np.all(vals <= end_us)


def test_poisson_mean_and_clip():
    lam = 4.2
    max_val = 63
    spec = CardinalitySpec(kind="poisson", params={"lam": lam, "max": max_val})
    vals = make_marginal(spec).ppf(_u())
    assert vals.dtype == np.int64
    assert abs(np.mean(vals) - lam) < 0.03 * lam
    assert np.all(vals >= 0) and np.all(vals <= max_val)


def test_fixed_cardinality():
    spec = CardinalitySpec(kind="fixed", params={"n": 7})
    vals = make_marginal(spec).ppf(_u())
    assert np.all(vals == 7)


def test_cardinality_uniform_int_clip():
    spec = CardinalitySpec(kind="uniform_int", params={"low": 0, "high": 100, "max": 10})
    vals = make_marginal(spec).ppf(_u())
    assert np.all(vals >= 0) and np.all(vals <= 10)


def test_beta_mean():
    a, b = 2.0, 5.0
    spec = DistributionSpec(kind="beta", params={"a": a, "b": b})
    vals = make_marginal(spec).ppf(_u())
    expected = a / (a + b)
    assert abs(np.mean(vals) - expected) < 0.03 * expected


def test_gamma_mean():
    shape, scale = 3.0, 2.0
    spec = DistributionSpec(kind="gamma", params={"shape": shape, "scale": scale})
    vals = make_marginal(spec).ppf(_u())
    expected = shape * scale
    assert abs(np.mean(vals) - expected) < 0.03 * expected


def test_determinism_same_spec_same_u():
    spec = DistributionSpec(kind="normal", params={"mean": 0.0, "std": 1.0})
    u = _u()
    v1 = make_marginal(spec).ppf(u)
    v2 = make_marginal(spec).ppf(u)
    np.testing.assert_array_equal(v1, v2)


def test_make_delay_ppf_matches_make_marginal():
    spec = DistributionSpec(kind="exponential", params={"rate": 1.0e-6})
    u = _u()
    v1 = make_marginal(spec).ppf(u)
    v2 = make_delay_ppf(spec)(u)
    np.testing.assert_array_equal(v1, v2)


def test_unknown_kind_raises():
    spec = DistributionSpec(kind="not_a_kind", params={})
    with pytest.raises(ValueError):
        make_marginal(spec)


def test_bernoulli_ppf_mean_and_values():
    spec = CardinalitySpec(kind="bernoulli", params={"p": 0.6})
    vals = make_marginal(spec).ppf(_u())
    assert set(np.unique(vals)) <= {0, 1}
    assert abs(np.mean(vals) - 0.6) < 0.01


def test_bernoulli_ppf_p_zero_all_zero():
    spec = CardinalitySpec(kind="bernoulli", params={"p": 0.0})
    vals = make_marginal(spec).ppf(_u())
    assert np.all(vals == 0)


def test_bernoulli_ppf_p_one_all_one():
    spec = CardinalitySpec(kind="bernoulli", params={"p": 1.0})
    vals = make_marginal(spec).ppf(_u())
    assert np.all(vals == 1)


# --------------------------------------------------------------------------
# zipf{a, n} (TASK CARD 13, docs/ARCHITECTURE.md §2)
# --------------------------------------------------------------------------


def test_zipf_ppf_range_and_rank0_frequency():
    a, n = 1.5, 50
    spec = DistributionSpec(kind="zipf", params={"a": a, "n": n})
    vals = make_marginal(spec).ppf(_u())

    assert vals.dtype == np.int64
    assert np.all(vals >= 0) and np.all(vals <= n - 1)

    j = np.arange(1, n + 1, dtype=np.float64)
    H = np.sum(j ** (-a))
    expected_rank0_freq = 1.0 / H
    freq0 = np.mean(vals == 0)
    assert abs(freq0 - expected_rank0_freq) < 0.03


def test_zipf_ppf_frequencies_non_increasing_first_five_ranks():
    a, n = 1.5, 50
    spec = DistributionSpec(kind="zipf", params={"a": a, "n": n})
    u = keyed_uniforms(1, "t.zipf_monotonic", np.arange(200_000, dtype=np.uint64))
    vals = make_marginal(spec).ppf(u)

    counts = [int(np.sum(vals == k)) for k in range(5)]
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))
