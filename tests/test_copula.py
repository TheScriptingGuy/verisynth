"""Acceptance tests for verisynth.copula. See docs/ARCHITECTURE.md §4."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtri, ndtr

from verisynth import kernels
from verisynth.copula import copula_uniforms, repair_correlation
from verisynth.metadata import CopulaSpec


def test_determinism():
    spec = CopulaSpec(name="g", columns=["a", "b"], correlation=[[1.0, 0.5], [0.5, 1.0]])
    keys = np.arange(500, dtype=np.uint64)
    r1 = copula_uniforms(42, "t", spec, keys)
    r2 = copula_uniforms(42, "t", spec, keys)
    assert r1.keys() == r2.keys()
    for c in spec.columns:
        np.testing.assert_array_equal(r1[c], r2[c])


def test_partition_invariance():
    spec = CopulaSpec(name="g", columns=["a", "b"], correlation=[[1.0, 0.4], [0.4, 1.0]])
    keys_all = np.arange(1000, dtype=np.uint64)
    keys_sub = np.arange(200, 300, dtype=np.uint64)

    r_all = copula_uniforms(7, "t", spec, keys_all)
    r_sub = copula_uniforms(7, "t", spec, keys_sub)

    for c in spec.columns:
        np.testing.assert_array_equal(r_all[c][200:300], r_sub[c])


def test_correlation_recovery_two_columns():
    rho = 0.7
    spec = CopulaSpec(name="g", columns=["a", "b"], correlation=[[1.0, rho], [rho, 1.0]])
    keys = np.arange(50_000, dtype=np.uint64)
    result = copula_uniforms(1, "t", spec, keys)

    za = ndtri(result["a"])
    zb = ndtri(result["b"])
    empirical_rho = np.corrcoef(za, zb)[0, 1]
    assert abs(empirical_rho - rho) < 0.02


def test_correlation_recovery_three_columns_mixed_signs():
    R = [
        [1.0, 0.6, -0.3],
        [0.6, 1.0, -0.2],
        [-0.3, -0.2, 1.0],
    ]
    spec = CopulaSpec(name="g3", columns=["a", "b", "c"], correlation=R)
    keys = np.arange(50_000, dtype=np.uint64)
    result = copula_uniforms(2, "t", spec, keys)

    z = np.stack([ndtri(result[c]) for c in spec.columns], axis=1)
    empirical = np.corrcoef(z, rowvar=False)

    np.testing.assert_allclose(empirical, np.asarray(R), atol=0.02)


def test_identity_correlation_independent_passthrough():
    spec = CopulaSpec(
        name="g",
        columns=["a", "b", "c"],
        correlation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    keys = np.arange(1000, dtype=np.uint64)
    result = copula_uniforms(99, "mytable", spec, keys)

    for c in spec.columns:
        namespace = f"mytable.__copula__.{spec.name}.{c}"
        direct = ndtr(kernels.inv_norm_cdf(kernels.keyed_uniforms(99, namespace, keys)))
        direct = np.clip(direct, 1e-15, 1 - 1e-15)
        np.testing.assert_allclose(result[c], direct, atol=1e-15)


def test_repair_correlation_non_pd_matrix():
    R = np.array(
        [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ]
    )
    # Confirm this input is indeed not PD.
    assert np.min(np.linalg.eigvalsh(R)) < 0

    repaired = repair_correlation(R)

    eigvals = np.linalg.eigvalsh(repaired)
    assert np.min(eigvals) > 0
    np.testing.assert_allclose(np.diag(repaired), 1.0, atol=1e-12)
    # Should not raise.
    np.linalg.cholesky(repaired)
    np.testing.assert_allclose(repaired, repaired.T, atol=1e-12)


def test_repair_correlation_valid_matrix_unchanged():
    R = np.array([[1.0, 0.55, 0.1], [0.55, 1.0, -0.2], [0.1, -0.2, 1.0]])
    repaired = repair_correlation(R)
    np.testing.assert_allclose(repaired, R, atol=1e-9)


def test_marginal_uniformity_preserved():
    rho = 0.7
    spec = CopulaSpec(name="g", columns=["a", "b"], correlation=[[1.0, rho], [rho, 1.0]])
    keys = np.arange(50_000, dtype=np.uint64)
    result = copula_uniforms(11, "t", spec, keys)

    for c in spec.columns:
        u = result[c]
        assert abs(np.mean(u) - 0.5) < 0.01
        for d in range(10):
            lo, hi = d / 10.0, (d + 1) / 10.0
            frac = np.mean((u >= lo) & (u < hi))
            assert abs(frac - 0.1) < 0.015


def test_outputs_strictly_inside_unit_interval():
    spec = CopulaSpec(name="g", columns=["a", "b"], correlation=[[1.0, 0.99], [0.99, 1.0]])
    keys = np.arange(10_000, dtype=np.uint64)
    result = copula_uniforms(3, "t", spec, keys)
    for c in spec.columns:
        assert np.all(result[c] > 0.0)
        assert np.all(result[c] < 1.0)


def test_empty_row_keys_does_not_crash():
    spec = CopulaSpec(name="g", columns=["a", "b"], correlation=[[1.0, 0.5], [0.5, 1.0]])
    keys = np.array([], dtype=np.uint64)
    result = copula_uniforms(1, "t", spec, keys)
    assert set(result.keys()) == {"a", "b"}
    for c in spec.columns:
        assert result[c].dtype == np.float64
        assert result[c].shape == (0,)
