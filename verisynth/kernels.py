"""Kernel dispatch: Rust extension when available, pure-numpy fallback.

See docs/ARCHITECTURE.md §1.3. Selects `verisynth_kernels` (Rust, via
PyO3) when importable; otherwise falls back to `verisynth._reference`.
Set env var `VERISYNTH_FORCE_REFERENCE=1` to force the reference backend
(useful for testing / when the Rust extension isn't built).
"""

from __future__ import annotations

import os

import numpy as np

from . import _reference

BACKEND: str

if os.environ.get("VERISYNTH_FORCE_REFERENCE") == "1":
    BACKEND = "reference"
    keyed_hash = _reference.keyed_hash
    keyed_uniforms = _reference.keyed_uniforms
    inv_norm_cdf = _reference.inv_norm_cdf
else:
    try:
        import verisynth_kernels as _rust

        BACKEND = "rust"

        def keyed_hash(seed: int, namespace: str, keys: np.ndarray, draw: int = 0) -> np.ndarray:
            keys = np.ascontiguousarray(keys, dtype=np.uint64)
            seed_u = int(seed) % (1 << 64)
            draw_u = int(draw) % (1 << 64)
            return _rust.keyed_hash(seed_u, namespace, keys, draw_u)

        def keyed_uniforms(seed: int, namespace: str, keys: np.ndarray, draw: int = 0) -> np.ndarray:
            keys = np.ascontiguousarray(keys, dtype=np.uint64)
            seed_u = int(seed) % (1 << 64)
            draw_u = int(draw) % (1 << 64)
            return _rust.keyed_uniforms(seed_u, namespace, keys, draw_u)

        def inv_norm_cdf(u: np.ndarray) -> np.ndarray:
            u = np.ascontiguousarray(u, dtype=np.float64)
            return _rust.inv_norm_cdf(u)

    except ImportError:
        BACKEND = "reference"
        keyed_hash = _reference.keyed_hash
        keyed_uniforms = _reference.keyed_uniforms
        inv_norm_cdf = _reference.inv_norm_cdf
