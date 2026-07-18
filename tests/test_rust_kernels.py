"""Acceptance tests for the Rust `verisynth_kernels` PyO3 extension.

See docs/ARCHITECTURE.md §1 and TASK CARD 3. Skips entirely when the
compiled wheel isn't installed, so the suite stays green on machines that
never built the extension.
"""

from __future__ import annotations

import importlib
import os
import time

import numpy as np
import pytest

verisynth_kernels = pytest.importorskip("verisynth_kernels")

from verisynth import _reference  # noqa: E402

SEEDS = [0, 1, 42, 2**63 + 11]
NAMESPACES = ["customers.age", "orders.__cardinality__", "t.c.__delay__"]
DRAWS = [0, 3]


def _keys():
    return np.random.default_rng(7).integers(0, 2**63, 4096).astype(np.uint64)


def _mask64(x: int) -> int:
    return x % (1 << 64)


def test_keyed_hash_bit_identical_to_reference():
    keys = _keys()
    for seed in SEEDS:
        for ns in NAMESPACES:
            for draw in DRAWS:
                got = verisynth_kernels.keyed_hash(_mask64(seed), ns, keys, _mask64(draw))
                want = _reference.keyed_hash(seed, ns, keys, draw)
                np.testing.assert_array_equal(
                    got, want, err_msg=f"seed={seed} ns={ns!r} draw={draw}"
                )
                assert np.asarray(got).dtype == np.uint64


def test_keyed_uniforms_bit_identical_to_reference():
    keys = _keys()
    for seed in SEEDS:
        for ns in NAMESPACES:
            for draw in DRAWS:
                got = verisynth_kernels.keyed_uniforms(_mask64(seed), ns, keys, _mask64(draw))
                want = _reference.keyed_uniforms(seed, ns, keys, draw)
                np.testing.assert_array_equal(
                    got, want, err_msg=f"seed={seed} ns={ns!r} draw={draw}"
                )
                assert np.asarray(got).dtype == np.float64


def test_inv_norm_cdf_matches_reference_within_1e12():
    keys = _keys()
    u_from_uniforms = _reference.keyed_uniforms(42, "customers.age", keys, 0)
    u_grid = np.linspace(1e-12, 1 - 1e-12, 20001)

    for u in (u_from_uniforms, u_grid):
        got = verisynth_kernels.inv_norm_cdf(u.astype(np.float64))
        want = _reference.inv_norm_cdf(u.astype(np.float64))
        max_abs_diff = np.max(np.abs(np.asarray(got) - want))
        assert max_abs_diff < 1e-12, f"max_abs_diff={max_abs_diff!r}"


def test_inv_norm_cdf_nan_outside_domain():
    u = np.array([0.0, 1.0, -0.5, 1.5, np.nan], dtype=np.float64)
    out = np.asarray(verisynth_kernels.inv_norm_cdf(u))
    assert np.all(np.isnan(out))


def test_dispatch_selects_rust_by_default_and_reference_when_forced(monkeypatch):
    monkeypatch.delenv("VERISYNTH_FORCE_REFERENCE", raising=False)
    import verisynth.kernels as kernels_mod

    importlib.reload(kernels_mod)
    try:
        assert kernels_mod.BACKEND == "rust"

        monkeypatch.setenv("VERISYNTH_FORCE_REFERENCE", "1")
        importlib.reload(kernels_mod)
        assert kernels_mod.BACKEND == "reference"
    finally:
        monkeypatch.delenv("VERISYNTH_FORCE_REFERENCE", raising=False)
        importlib.reload(kernels_mod)
        assert kernels_mod.BACKEND == "rust"


def test_benchmark_rust_vs_reference(capsys):
    keys = np.arange(1_000_000, dtype=np.uint64)

    t0 = time.perf_counter()
    _reference.keyed_uniforms(42, "bench.column", keys, 0)
    t_reference = time.perf_counter() - t0

    t0 = time.perf_counter()
    verisynth_kernels.keyed_uniforms(42, "bench.column", keys, 0)
    t_rust = time.perf_counter() - t0

    with capsys.disabled():
        print(
            f"\n[benchmark] keyed_uniforms on 1e6 keys: "
            f"rust={t_rust * 1e3:.3f} ms, reference={t_reference * 1e3:.3f} ms, "
            f"speedup={t_reference / t_rust if t_rust else float('inf'):.1f}x"
        )
