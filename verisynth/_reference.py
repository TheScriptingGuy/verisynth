"""Pure-numpy reference kernels for deterministic keyed generation.

Implements docs/ARCHITECTURE.md §1.1-1.2 exactly. This module is the
normative "ground truth" backend: the Rust extension (when present) must
be bit-identical for `keyed_hash`/`keyed_uniforms` and agree with
`inv_norm_cdf` within 1e-12 absolute (see verisynth/kernels.py).
"""

from __future__ import annotations

import numpy as np

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX_C1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_C2 = np.uint64(0x94D049BB133111EB)
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64_PY = (1 << 64) - 1


def fnv1a64(s: str) -> int:
    """FNV-1a 64-bit hash over the UTF-8 bytes of ``s``, mod 2**64."""
    h = _FNV_OFFSET
    for b in s.encode("utf-8"):
        h = (h ^ b) * _FNV_PRIME
        h &= _MASK64_PY
    return h


def _mix64(z: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer, vectorized over a uint64 array."""
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * _MIX_C1
        z = (z ^ (z >> np.uint64(27))) * _MIX_C2
        z = z ^ (z >> np.uint64(31))
    return z


def keyed_hash(seed: int, namespace: str, keys: np.ndarray, draw: int = 0) -> np.ndarray:
    """Vectorized cell hash chain; returns uint64 array shaped like ``keys``."""
    keys = np.asarray(keys, dtype=np.uint64)
    seed_u = np.uint64(int(seed) % (1 << 64))
    draw_u = np.uint64(int(draw) % (1 << 64))
    ns_hash = np.uint64(fnv1a64(namespace))

    with np.errstate(over="ignore"):
        h = _mix64(seed_u ^ _GOLDEN)
        h = _mix64(h ^ ns_hash)
        h = _mix64(keys ^ h)
        h = _mix64(h ^ draw_u)
    return h


def keyed_uniforms(seed: int, namespace: str, keys: np.ndarray, draw: int = 0) -> np.ndarray:
    """Vectorized uniforms in the open interval (0, 1), float64."""
    h = keyed_hash(seed, namespace, keys, draw)
    with np.errstate(over="ignore"):
        top53 = (h >> np.uint64(11)).astype(np.float64)
    return (top53 + 0.5) * 2.0**-53


# --------------------------------------------------------------------------
# Acklam's rational approximation of the standard normal inverse CDF.
# --------------------------------------------------------------------------

_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425
_P_HIGH = 1 - _P_LOW


def inv_norm_cdf(u: np.ndarray) -> np.ndarray:
    """Acklam's approximation of Phi^-1, vectorized. NaN outside (0, 1)."""
    u = np.asarray(u, dtype=np.float64)
    domain = (u > 0.0) & (u < 1.0)

    out = np.full_like(u, np.nan, dtype=np.float64)

    low_mask = domain & (u < _P_LOW)
    high_mask = domain & (u > _P_HIGH)
    mid_mask = domain & ~low_mask & ~high_mask

    # Low tail.
    if np.any(low_mask):
        q = np.sqrt(-2.0 * np.log(u[low_mask]))
        num = ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
        den = (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        out[low_mask] = num / den

    # Central region.
    if np.any(mid_mask):
        q = u[mid_mask] - 0.5
        r = q * q
        num = (
            ((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]
        ) * q
        den = (
            ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        )
        out[mid_mask] = num / den

    # High tail.
    if np.any(high_mask):
        q = np.sqrt(-2.0 * np.log(1.0 - u[high_mask]))
        num = ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
        den = (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        out[high_mask] = -num / den

    return out
