"""Gaussian copula: correlated uniforms per row. See docs/ARCHITECTURE.md §4.

Deterministic given (seed, table, copula spec, row_keys): each latent
component is drawn from the keyed RNG kernels, correlated via a Cholesky
factor of a repaired correlation matrix, then mapped back to uniforms via
the standard normal CDF.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from . import kernels
from .metadata import CopulaSpec

_EIGENVALUE_FLOOR = 1e-6
_CLIP_LO = 1e-15
_CLIP_HI = 1.0 - 1e-15


def repair_correlation(R: np.ndarray) -> np.ndarray:
    """Symmetrize, eigenvalue-clip, and renormalize ``R`` to a valid
    symmetric positive-definite correlation matrix with unit diagonal.

    See docs/ARCHITECTURE.md §4 step 1.
    """
    R = np.asarray(R, dtype=np.float64)
    R = (R + R.T) / 2.0

    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.clip(eigvals, _EIGENVALUE_FLOOR, None)
    R = (eigvecs * eigvals) @ eigvecs.T

    d = np.sqrt(np.diag(R))
    R = R / d[:, None] / d[None, :]

    return R


def copula_uniforms(
    seed: int, table: str, spec: CopulaSpec, row_keys: np.ndarray
) -> dict[str, np.ndarray]:
    """Correlated uniforms for each column in ``spec``, keyed by ``row_keys``.

    See docs/ARCHITECTURE.md §4.
    """
    row_keys = np.asarray(row_keys, dtype=np.uint64)
    n = row_keys.shape[0]
    k = len(spec.columns)

    R = repair_correlation(np.asarray(spec.correlation, dtype=np.float64))
    L = np.linalg.cholesky(R)

    E = np.empty((n, k), dtype=np.float64)
    for j, column in enumerate(spec.columns):
        namespace = f"{table}.__copula__.{spec.name}.{column}"
        E[:, j] = kernels.inv_norm_cdf(kernels.keyed_uniforms(seed, namespace, row_keys))

    Z = E @ L.T
    U = ndtr(Z)
    U = np.clip(U, _CLIP_LO, _CLIP_HI)

    return {column: U[:, j] for j, column in enumerate(spec.columns)}
