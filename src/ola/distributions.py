"""Distributions over the highest competing bid ``m_t``.

In the stochastic environment of Requirement 1 the highest competing bid is
drawn i.i.d. from a distribution ``D`` supported on ``[0, 1]``.  Each
distribution exposes:

* ``sample(n, rng)`` -- draw ``n`` realizations of ``m`` (clipped to ``[0, 1]``);
* ``cdf(b)`` -- ``P(m <= b)``, i.e. the probability of winning when bidding ``b``
  (since the advertiser wins iff ``b >= m``).

``cdf`` is what makes the *true* expected utility/cost (and hence the
clairvoyant baselines) computable in closed form:

    E[f(b)] = (v - b) * P(m <= b) = (v - b) * cdf(b)
    E[c(b)] = b       * P(m <= b) = b       * cdf(b)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


class CompetingBidDistribution:
    """Base class for distributions over the highest competing bid."""

    name: str = "base"

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def cdf(self, b: np.ndarray | float) -> np.ndarray:
        """Return ``P(m <= b)`` evaluated (element-wise) at ``b``."""
        raise NotImplementedError

    def ppf(self, u: np.ndarray | float) -> np.ndarray:
        """Inverse CDF (quantile function), used for copula-based sampling."""
        raise NotImplementedError

    def to_config(self) -> dict[str, Any]:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    @staticmethod
    def from_config(config: dict[str, Any]) -> "CompetingBidDistribution":
        kind = config["type"]
        params = {k: v for k, v in config.items() if k != "type"}
        if kind == "uniform":
            return UniformDistribution(**params)
        if kind == "truncnorm":
            return TruncatedNormalDistribution(**params)
        if kind == "beta":
            return BetaDistribution(**params)
        if kind == "bimodal":
            return BimodalDistribution(**params)
        raise ValueError(f"Unknown distribution type: {kind!r}")


class UniformDistribution(CompetingBidDistribution):
    """Uniform distribution on ``[low, high] subseteq [0, 1]``."""

    name = "uniform"

    def __init__(self, low: float = 0.0, high: float = 1.0):
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("require 0 <= low < high <= 1")
        self.low = float(low)
        self.high = float(high)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=n)

    def cdf(self, b: np.ndarray | float) -> np.ndarray:
        b = np.asarray(b, dtype=float)
        return np.clip((b - self.low) / (self.high - self.low), 0.0, 1.0)

    def ppf(self, u: np.ndarray | float) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return self.low + np.clip(u, 0.0, 1.0) * (self.high - self.low)

    def to_config(self) -> dict[str, Any]:
        return {"type": self.name, "low": self.low, "high": self.high}


class TruncatedNormalDistribution(CompetingBidDistribution):
    """Normal distribution with ``mean``/``std`` truncated to ``[low, high]``."""

    name = "truncnorm"

    def __init__(
        self,
        mean: float = 0.5,
        std: float = 0.15,
        low: float = 0.0,
        high: float = 1.0,
    ):
        if std <= 0:
            raise ValueError("std must be positive")
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("require 0 <= low < high <= 1")
        self.mean = float(mean)
        self.std = float(std)
        self.low = float(low)
        self.high = float(high)
        a, b = (self.low - self.mean) / self.std, (self.high - self.mean) / self.std
        self._rv = stats.truncnorm(a, b, loc=self.mean, scale=self.std)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self._rv.rvs(size=n, random_state=rng)

    def cdf(self, b: np.ndarray | float) -> np.ndarray:
        b = np.asarray(b, dtype=float)
        return self._rv.cdf(b)

    def ppf(self, u: np.ndarray | float) -> np.ndarray:
        return np.clip(self._rv.ppf(np.clip(u, 0.0, 1.0)), 0.0, 1.0)

    def to_config(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "mean": self.mean,
            "std": self.std,
            "low": self.low,
            "high": self.high,
        }


class BetaDistribution(CompetingBidDistribution):
    """Beta(a, b) distribution, naturally supported on ``[0, 1]``."""

    name = "beta"

    def __init__(self, a: float = 2.0, b: float = 5.0):
        if a <= 0 or b <= 0:
            raise ValueError("a and b must be positive")
        self.a = float(a)
        self.b = float(b)
        self._rv = stats.beta(self.a, self.b)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self._rv.rvs(size=n, random_state=rng)

    def cdf(self, b: np.ndarray | float) -> np.ndarray:
        b = np.asarray(b, dtype=float)
        return self._rv.cdf(np.clip(b, 0.0, 1.0))

    def ppf(self, u: np.ndarray | float) -> np.ndarray:
        return np.clip(self._rv.ppf(np.clip(u, 0.0, 1.0)), 0.0, 1.0)

    def to_config(self) -> dict[str, Any]:
        return {"type": self.name, "a": self.a, "b": self.b}


class BimodalDistribution(CompetingBidDistribution):
    """Mixture of two truncated normals.

    Models a market with two populations of competitors: a "cheap" mode and an
    "expensive" mode.  Useful to stress budget pacing, because winning the
    expensive mode quickly drains the budget.
    """

    name = "bimodal"

    def __init__(
        self,
        weight_low: float = 0.6,
        mean_low: float = 0.2,
        std_low: float = 0.07,
        mean_high: float = 0.75,
        std_high: float = 0.07,
    ):
        if not 0.0 <= weight_low <= 1.0:
            raise ValueError("weight_low must be in [0, 1]")
        self.weight_low = float(weight_low)
        self._low = TruncatedNormalDistribution(mean_low, std_low)
        self._high = TruncatedNormalDistribution(mean_high, std_high)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        pick_low = rng.random(n) < self.weight_low
        low = self._low.sample(n, rng)
        high = self._high.sample(n, rng)
        return np.where(pick_low, low, high)

    def cdf(self, b: np.ndarray | float) -> np.ndarray:
        b = np.asarray(b, dtype=float)
        return self.weight_low * self._low.cdf(b) + (1.0 - self.weight_low) * self._high.cdf(b)

    def ppf(self, u: np.ndarray | float) -> np.ndarray:
        # The mixture CDF has no closed-form inverse; invert numerically on a
        # fine grid (the CDF is monotone, so linear interpolation is accurate).
        u = np.asarray(u, dtype=float)
        grid = np.linspace(0.0, 1.0, 4001)
        cdf_grid = self.cdf(grid)
        return np.interp(np.clip(u, 0.0, 1.0), cdf_grid, grid)

    def to_config(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "weight_low": self.weight_low,
            "mean_low": self._low.mean,
            "std_low": self._low.std,
            "mean_high": self._high.mean,
            "std_high": self._high.std,
        }


class JointCompetingBidDistribution:
    """Joint distribution over the highest competing bids of several campaigns.

    Used by the multi-campaign (Requirement 2) environment.  It is built from a
    list of marginal distributions (one per campaign) and an optional
    equicorrelation parameter ``rho`` in ``[0, 1)`` realized through a Gaussian
    copula:

    * draw a correlated Gaussian vector ``z ~ N(0, R)`` with
      ``R = (1 - rho) I + rho 11^T``;
    * map to uniforms ``u = Phi(z)``;
    * apply each campaign's quantile function ``m_i = F_i^{-1}(u_i)``.

    The correlation only couples the *realizations* across campaigns; every
    marginal -- and therefore every per-arm expected utility/cost used by the
    algorithm and the clairvoyant baseline -- is exactly the campaign's own
    distribution.  With ``rho = 0`` the campaigns are independent.
    """

    def __init__(self, marginals: list[CompetingBidDistribution], correlation: float = 0.0):
        if not -0.0 <= correlation < 1.0:
            raise ValueError("correlation must be in [0, 1)")
        self.marginals = list(marginals)
        self.n = len(self.marginals)
        self.correlation = float(correlation)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return an ``(n, N)`` array of jointly drawn competing bids."""
        if self.correlation <= 0.0 or self.n == 1:
            cols = [m.sample(n, rng) for m in self.marginals]
            return np.column_stack(cols)

        from scipy.stats import norm

        cov = (1.0 - self.correlation) * np.eye(self.n) + self.correlation * np.ones((self.n, self.n))
        z = rng.multivariate_normal(np.zeros(self.n), cov, size=n)
        u = norm.cdf(z)
        cols = [self.marginals[i].ppf(u[:, i]) for i in range(self.n)]
        return np.column_stack(cols)

    def marginal_cdf(self, i: int, b: np.ndarray | float) -> np.ndarray:
        return self.marginals[i].cdf(b)
