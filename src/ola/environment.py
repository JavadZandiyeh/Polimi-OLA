"""Environments and the scenario configuration that defines them.

A :class:`Scenario` is the serializable description of a problem instance
(valuation, discrete bid set, horizon, budget and the distribution over the
highest competing bid).  It is the on-disk "dataset" produced by
``data/generate_datasets.py``.

:class:`StochasticSingleCampaignEnv` is the Requirement 1 environment: at each
round it draws ``m_t ~ D`` and returns the first-price utility/cost of the bid
played (bandit feedback).  It deliberately does **not** track the budget --
budget bookkeeping and the termination rule live in
:mod:`ola.simulation`, so that they are applied identically to every algorithm.

:class:`MultiCampaignScenario` / :class:`MultiCampaignStochasticEnv` are the
Requirement 2 multi-campaign counterparts: at each round a *joint* vector of
highest competing bids is drawn, the agency bids in a (conflict-feasible) set of
campaigns and observes per-campaign (semi-bandit) feedback.

:class:`AdversarialSingleCampaignEnv` is a stub for the adversarial requirement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .auction import cost, utility
from .distributions import CompetingBidDistribution, JointCompetingBidDistribution


@dataclass
class Scenario:
    """Serializable description of a single-campaign problem instance."""

    name: str
    description: str
    valuation: float
    bids: list[float]
    horizon: int
    budget: float
    distribution: dict[str, Any]
    seed: int = 0

    @property
    def rho(self) -> float:
        """Per-round budget ``rho = B / T``."""
        return self.budget / self.horizon

    def bid_array(self) -> np.ndarray:
        return np.asarray(self.bids, dtype=float)

    def make_distribution(self) -> CompetingBidDistribution:
        return CompetingBidDistribution.from_config(self.distribution)

    # --------------------------- (de)serialization -------------------- #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class Outcome:
    """Result of a single auction round (what the environment reveals)."""

    won: bool
    utility: float
    cost: float
    competing_bid: float  # m_t -- available for logging, NOT for bandit agents


class StochasticSingleCampaignEnv:
    """Requirement 1 environment: i.i.d. stochastic highest competing bid."""

    def __init__(self, scenario: Scenario, rng: np.random.Generator):
        self.scenario = scenario
        self.valuation = float(scenario.valuation)
        self.bids = scenario.bid_array()
        self.horizon = int(scenario.horizon)
        self.distribution = scenario.make_distribution()
        self.rng = rng
        self._t = 0

    def step(self, bid: float) -> Outcome:
        """Play ``bid`` against a freshly sampled ``m_t`` and return the outcome."""
        m = float(self.distribution.sample(1, self.rng)[0])
        won = bid >= m
        return Outcome(
            won=won,
            utility=float(utility(self.valuation, bid, m)),
            cost=float(cost(bid, m)),
            competing_bid=m,
        )

    def sample_competing_bids(self, n: int) -> np.ndarray:
        """Draw ``n`` i.i.d. realizations of ``m`` (used to materialize datasets)."""
        return self.distribution.sample(n, self.rng)


# --------------------------------------------------------------------------- #
# Stubs for later requirements.
# --------------------------------------------------------------------------- #
class AdversarialSingleCampaignEnv:
    """Stub: single campaign with an adversarial sequence of ``m_t``.

    To be implemented for the adversarial requirement, where ``m_t`` is an
    arbitrary (possibly worst-case) sequence rather than i.i.d. samples, and the
    relevant algorithm is primal-dual pacing with a regret minimizer
    (see :mod:`ola.algorithms.pacing`).
    """

    def __init__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "AdversarialSingleCampaignEnv is a stub for a later requirement."
        )


@dataclass
class MultiCampaignScenario:
    """Serializable description of a multi-campaign (Requirement 2) instance.

    ``conflict_edges`` lists pairs of mutually exclusive campaigns (the conflict
    graph from the project overview); an empty list is the pure multi-campaign
    setting.  ``correlation`` is the Gaussian-copula equicorrelation of the
    joint competing-bid distribution (``0`` = independent campaigns).
    """

    name: str
    description: str
    valuations: list[float]
    bids: list[float]
    horizon: int
    budget: float
    distributions: list[dict[str, Any]]
    conflict_edges: list[list[int]] = field(default_factory=list)
    correlation: float = 0.0
    seed: int = 0

    @property
    def n_campaigns(self) -> int:
        return len(self.valuations)

    @property
    def rho(self) -> float:
        return self.budget / self.horizon

    def bid_array(self) -> np.ndarray:
        return np.asarray(self.bids, dtype=float)

    def valuation_array(self) -> np.ndarray:
        return np.asarray(self.valuations, dtype=float)

    def edge_list(self) -> list[tuple[int, int]]:
        return [(int(u), int(v)) for u, v in self.conflict_edges]

    def make_marginals(self) -> list[CompetingBidDistribution]:
        return [CompetingBidDistribution.from_config(d) for d in self.distributions]

    def make_joint_distribution(self) -> JointCompetingBidDistribution:
        return JointCompetingBidDistribution(self.make_marginals(), self.correlation)

    # --------------------------- (de)serialization -------------------- #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiCampaignScenario":
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> "MultiCampaignScenario":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class MultiOutcome:
    """Result of one multi-campaign round (semi-bandit feedback)."""

    # campaign index -> (bid_index, won, utility, cost)
    per_campaign: dict[int, tuple[int, bool, float, float]]
    total_utility: float
    total_cost: float
    competing_bids: np.ndarray  # m vector (all campaigns); only active observed


class MultiCampaignStochasticEnv:
    """Requirement 2 environment: ``N`` simultaneous first-price campaigns.

    Each round a joint vector ``m = (m_1, ..., m_N)`` is sampled from the joint
    distribution.  The agency plays a set of (campaign -> bid) actions; for every
    played campaign ``i`` with bid ``b`` it wins iff ``b >= m_i``, gaining utility
    ``v_i - b`` and paying ``b``.  Feedback is semi-bandit: only the played
    campaigns' outcomes are revealed.  As in the single-campaign env, budget
    bookkeeping lives in :mod:`ola.simulation`.
    """

    def __init__(self, scenario: MultiCampaignScenario, rng: np.random.Generator):
        self.scenario = scenario
        self.valuations = scenario.valuation_array()
        self.bids = scenario.bid_array()
        self.horizon = int(scenario.horizon)
        self.n_campaigns = scenario.n_campaigns
        self.distribution = scenario.make_joint_distribution()
        self.rng = rng
        # Pre-sample the whole joint sequence (vectorized): a single draw is much
        # faster than per-round sampling, especially under a Gaussian copula.
        self._samples = self.distribution.sample(self.horizon, rng)
        self._t = 0

    def step(self, actions: dict[int, int]) -> MultiOutcome:
        """Play ``actions`` (campaign -> bid index) against the round's ``m`` vector."""
        m = self._samples[self._t]
        self._t += 1
        per_campaign: dict[int, tuple[int, bool, float, float]] = {}
        total_u = 0.0
        total_c = 0.0
        for i, bid_index in actions.items():
            b = float(self.bids[bid_index])
            won = b >= m[i]
            u = float(utility(self.valuations[i], b, m[i]))
            c = float(cost(b, m[i]))
            per_campaign[i] = (bid_index, bool(won), u, c)
            total_u += u
            total_c += c
        return MultiOutcome(per_campaign, total_u, total_c, m)

    def sample_competing_bids(self, n: int) -> np.ndarray:
        """Draw ``n`` joint realizations (shape ``(n, N)``) to materialize data."""
        return self.distribution.sample(n, self.rng)

    def competing_bid_sequence(self) -> np.ndarray:
        """The pre-sampled ``(T, N)`` competing-bid sequence (for hindsight)."""
        return self._samples


# --------------------------------------------------------------------------- #
# Non-stationary multi-campaign environment (Requirement 3).
# --------------------------------------------------------------------------- #
def nonstationary_sequence(
    schedule: dict[str, Any],
    n_campaigns: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Materialize a ``(T, N)`` *non-stochastic* sequence of competing bids.

    The sequence is the "dataset" of a Requirement 3 non-stationary environment:
    a fixed (seed-determined) trajectory whose underlying distribution changes
    over time.  Three regimes are supported, matching the course taxonomy of
    non-stationary environments (Part 10):

    * ``"abrupt"`` -- piecewise-stationary.  The horizon is split into phases of
      length ``phase_length``; in phase ``p`` campaign ``i`` draws i.i.d. from
      ``campaign_phase_dists[i][p]`` (cycled if there are more phases than
      configs).  Models a new bidder entering / leaving the auctions.
    * ``"smooth"`` -- the mean of each campaign's truncated-normal competing bid
      drifts continuously, ``mu_i(t) = center_i + amp_i * sin(2 pi f_i t / T +
      phase_i)``.  Models slowly fluctuating competition.
    * ``"fast"`` -- "a distribution that changes quickly over time": every
      ``block_length`` rounds each campaign gets a fresh mean drawn uniformly in
      ``[low, high]`` and samples a tight truncated normal around it.  This is the
      highly non-stationary regime emphasized by Requirement 3.

    All draws are clipped to ``[0, 1]``.
    """
    kind = schedule["kind"]
    m = np.zeros((horizon, n_campaigns))

    if kind == "abrupt":
        phase_len = int(schedule["phase_length"])
        per_campaign = schedule["campaign_phase_dists"]
        for i in range(n_campaigns):
            configs = per_campaign[i]
            for t0 in range(0, horizon, phase_len):
                t1 = min(t0 + phase_len, horizon)
                phase_idx = (t0 // phase_len) % len(configs)
                dist = CompetingBidDistribution.from_config(configs[phase_idx])
                m[t0:t1, i] = dist.sample(t1 - t0, rng)

    elif kind == "smooth":
        params = schedule["smooth"]
        t = np.arange(horizon)
        for i in range(n_campaigns):
            p = params[i]
            mu = p["center"] + p["amp"] * np.sin(
                2.0 * np.pi * p["freq"] * t / horizon + p.get("phase", 0.0)
            )
            mu = np.clip(mu, 0.02, 0.98)
            std = float(p["std"])
            draws = mu + std * rng.standard_normal(horizon)
            m[:, i] = draws

    elif kind == "fast":
        block = int(schedule["block_length"])
        low, high = float(schedule["low"]), float(schedule["high"])
        std = float(schedule["std"])
        for t0 in range(0, horizon, block):
            t1 = min(t0 + block, horizon)
            means = rng.uniform(low, high, size=n_campaigns)
            for i in range(n_campaigns):
                m[t0:t1, i] = means[i] + std * rng.standard_normal(t1 - t0)

    else:
        raise ValueError(f"Unknown non-stationary schedule kind: {kind!r}")

    return np.clip(m, 0.0, 1.0)


@dataclass
class NonStationaryMultiCampaignScenario:
    """Serializable description of a Requirement 3 non-stationary instance.

    Shares the multi-campaign structure (valuations, discrete bids, global
    budget, conflict graph) with :class:`MultiCampaignScenario`, but instead of a
    stationary joint distribution it carries a ``schedule`` describing how the
    per-campaign competing-bid distribution changes over time (see
    :func:`nonstationary_sequence`).  The actual competing-bid trajectory is fixed
    (a *non-stochastic* sequence) and reproducible from ``seed``.
    """

    name: str
    description: str
    valuations: list[float]
    bids: list[float]
    horizon: int
    budget: float
    schedule: dict[str, Any]
    conflict_edges: list[list[int]] = field(default_factory=list)
    seed: int = 0

    @property
    def n_campaigns(self) -> int:
        return len(self.valuations)

    @property
    def rho(self) -> float:
        return self.budget / self.horizon

    def bid_array(self) -> np.ndarray:
        return np.asarray(self.bids, dtype=float)

    def valuation_array(self) -> np.ndarray:
        return np.asarray(self.valuations, dtype=float)

    def edge_list(self) -> list[tuple[int, int]]:
        return [(int(u), int(v)) for u, v in self.conflict_edges]

    def make_sequence(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Materialize the fixed ``(T, N)`` competing-bid sequence."""
        if rng is None:
            rng = np.random.default_rng(self.seed)
        return nonstationary_sequence(self.schedule, self.n_campaigns, self.horizon, rng)

    # --------------------------- (de)serialization -------------------- #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NonStationaryMultiCampaignScenario":
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> "NonStationaryMultiCampaignScenario":
        return cls.from_dict(json.loads(Path(path).read_text()))


class NonStationaryMultiCampaignEnv:
    """Requirement 3 environment: ``N`` campaigns with a non-stationary ``m``.

    Mirrors :class:`MultiCampaignStochasticEnv` (same first-price mechanics, same
    semi-bandit accounting of *played* campaigns) but replays the fixed
    non-stochastic sequence produced by the scenario's ``schedule`` instead of
    i.i.d. joint draws.  Crucially, every round's full competing-bid vector is
    exposed through :class:`MultiOutcome.competing_bids`, because Requirement 3
    grants **full feedback** (the agency observes ``m_i`` for every campaign).
    """

    def __init__(
        self,
        scenario: NonStationaryMultiCampaignScenario,
        rng: np.random.Generator | None = None,
    ):
        self.scenario = scenario
        self.valuations = scenario.valuation_array()
        self.bids = scenario.bid_array()
        self.horizon = int(scenario.horizon)
        self.n_campaigns = scenario.n_campaigns
        # The competing-bid sequence is fixed (adversarial / non-stochastic); it
        # is derived from the scenario seed and does NOT depend on ``rng`` (which
        # only governs any algorithm-side randomization elsewhere).
        self._samples = scenario.make_sequence()
        self.rng = rng
        self._t = 0

    def step(self, actions: dict[int, int]) -> MultiOutcome:
        m = self._samples[self._t]
        self._t += 1
        per_campaign: dict[int, tuple[int, bool, float, float]] = {}
        total_u = 0.0
        total_c = 0.0
        for i, bid_index in actions.items():
            b = float(self.bids[bid_index])
            won = b >= m[i]
            u = float(utility(self.valuations[i], b, m[i]))
            c = float(cost(b, m[i]))
            per_campaign[i] = (bid_index, bool(won), u, c)
            total_u += u
            total_c += c
        return MultiOutcome(per_campaign, total_u, total_c, m)

    def competing_bid_sequence(self) -> np.ndarray:
        """The fixed ``(T, N)`` competing-bid sequence (for hindsight)."""
        return self._samples
