"""Laplace mechanism primitives and epsilon-budget splitting.

See docs/ARCHITECTURE.md §7 (normative). DP noise here uses its own
``numpy.random.Generator`` — it is *not* part of the deterministic keyed
generation pipeline (``verisynth.kernels``).
"""

from __future__ import annotations

import numpy as np


def split_epsilon(total: float, k: int) -> float:
    """Split a total epsilon budget evenly across ``k`` released statistics."""
    if total <= 0:
        raise ValueError(f"total epsilon must be > 0 (got {total!r})")
    if k < 1:
        raise ValueError(f"k must be >= 1 (got {k!r})")
    return total / k


def laplace_value(
    value: float, sensitivity: float, epsilon: float, rng: np.random.Generator
) -> float:
    """Add Laplace(0, sensitivity/epsilon) noise to ``value``."""
    return value + rng.laplace(0.0, sensitivity / epsilon)


def noisy_counts(
    counts: np.ndarray, epsilon: float, rng: np.random.Generator
) -> np.ndarray:
    """Add Laplace(0, 1/epsilon) noise to each count (sensitivity 1 per count),
    clamped to be non-negative."""
    counts = np.asarray(counts, dtype=np.float64)
    noisy = counts + rng.laplace(0.0, 1.0 / epsilon, size=counts.shape)
    return np.maximum(0.0, noisy)
