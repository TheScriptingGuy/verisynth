"""Inverse-CDF marginal samplers per column/cardinality distribution spec.

See docs/ARCHITECTURE.md §2 for the distribution kinds. Each sampler's
``.ppf(u)`` takes an array of float64 uniforms in the open interval
(0, 1) (as produced by ``verisynth.kernels.keyed_uniforms``, optionally
post-processed by the Gaussian copula) and returns the mapped value
array.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
from scipy import stats

from . import kernels
from .metadata import DistributionSpec


@runtime_checkable
class Marginal(Protocol):
    def ppf(self, u: np.ndarray) -> np.ndarray: ...


@dataclass
class _CategoricalMarginal:
    categories: np.ndarray  # dtype=object
    cumprobs: np.ndarray  # dtype=float64

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        idx = np.searchsorted(self.cumprobs, u, side="right")
        idx = np.clip(idx, 0, len(self.categories) - 1)
        return self.categories[idx]


@dataclass
class _NormalMarginal:
    mean: float
    std: float

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return self.mean + self.std * kernels.inv_norm_cdf(np.asarray(u, dtype=np.float64))


@dataclass
class _LognormalMarginal:
    mu: float
    sigma: float

    def ppf(self, u: np.ndarray) -> np.ndarray:
        z = kernels.inv_norm_cdf(np.asarray(u, dtype=np.float64))
        return np.exp(self.mu + self.sigma * z)


@dataclass
class _UniformMarginal:
    low: float
    high: float

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        return self.low + (self.high - self.low) * u


@dataclass
class _ExponentialMarginal:
    rate: float

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        return -np.log1p(-u) / self.rate


@dataclass
class _GammaMarginal:
    shape: float
    scale: float

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        return stats.gamma.ppf(u, a=self.shape, scale=self.scale)


@dataclass
class _BetaMarginal:
    a: float
    b: float

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        return stats.beta.ppf(u, self.a, self.b)


@dataclass
class _UniformIntMarginal:
    low: int
    high: int

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        n = self.high - self.low + 1
        out = np.floor(self.low + n * u).astype(np.int64)
        return np.clip(out, self.low, self.high)


@dataclass
class _DatetimeUniformMarginal:
    start_us: int
    end_us: int

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        out = self.start_us + (self.end_us - self.start_us) * u
        return out.astype(np.int64)


@dataclass
class _PoissonMarginal:
    lam: float
    max: int

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        out = stats.poisson.ppf(u, mu=self.lam).astype(np.int64)
        return np.clip(out, 0, self.max)


@dataclass
class _FixedMarginal:
    n: int

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        return np.full(u.shape, self.n, dtype=np.int64)


@dataclass
class _ClippedUniformIntMarginal:
    low: int
    high: int
    max: int

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        n = self.high - self.low + 1
        out = np.floor(self.low + n * u).astype(np.int64)
        out = np.clip(out, self.low, self.high)
        return np.clip(out, 0, self.max)


def _parse_epoch_us(s: str) -> int:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)


def make_marginal(spec: DistributionSpec) -> Marginal:
    """Map a ``DistributionSpec`` (or cardinality spec) to a ``Marginal``."""
    kind = spec.kind
    p: dict[str, Any] = spec.params

    if kind == "categorical":
        categories = np.asarray(p["categories"], dtype=object)
        probs = np.asarray(p["probs"], dtype=np.float64)
        cumprobs = np.cumsum(probs)
        return _CategoricalMarginal(categories=categories, cumprobs=cumprobs)
    if kind == "normal":
        return _NormalMarginal(mean=float(p["mean"]), std=float(p["std"]))
    if kind == "lognormal":
        return _LognormalMarginal(mu=float(p["mu"]), sigma=float(p["sigma"]))
    if kind == "uniform":
        return _UniformMarginal(low=float(p["low"]), high=float(p["high"]))
    if kind == "exponential":
        return _ExponentialMarginal(rate=float(p["rate"]))
    if kind == "gamma":
        return _GammaMarginal(shape=float(p["shape"]), scale=float(p["scale"]))
    if kind == "beta":
        return _BetaMarginal(a=float(p["a"]), b=float(p["b"]))
    if kind == "uniform_int":
        if "max" in p:
            return _ClippedUniformIntMarginal(
                low=int(p["low"]), high=int(p["high"]), max=int(p["max"])
            )
        return _UniformIntMarginal(low=int(p["low"]), high=int(p["high"]))
    if kind == "datetime_uniform":
        start_us = _parse_epoch_us(p["start"])
        end_us = _parse_epoch_us(p["end"])
        return _DatetimeUniformMarginal(start_us=start_us, end_us=end_us)
    if kind == "poisson":
        return _PoissonMarginal(lam=float(p["lam"]), max=int(p["max"]))
    if kind == "fixed":
        return _FixedMarginal(n=int(p["n"]))

    raise ValueError(f"unknown distribution kind {kind!r}")


def make_delay_ppf(spec: DistributionSpec) -> Callable[[np.ndarray], np.ndarray]:
    """Convenience alias: same as ``make_marginal(spec).ppf``."""
    return make_marginal(spec).ppf
