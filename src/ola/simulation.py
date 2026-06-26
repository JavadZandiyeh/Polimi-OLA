"""Episode runner.

Plays a bidder against an environment for ``T`` rounds while enforcing the
budget.  Budget logic is centralized here (not in the bidders) so that the
budget-ignoring Algorithm A and the budget-aware Algorithm B are evaluated under
exactly the same rule:

* the bidder stops participating as soon as the remaining budget can no longer
  cover the largest possible cost (the maximum bid) -- mirroring the
  "if B < 1 then terminate" rule of the course (bids live in ``[0, 1]`` so the
  maximum cost is the maximum bid);
* once stopped, the remaining rounds earn zero utility and cost.

All per-round quantities are returned as arrays of length ``T`` for plotting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .algorithms.base import Bidder
from .algorithms.combinatorial_ucb import CombinatorialUCBBidder
from .algorithms.primal_dual import PrimalDualBidder
from .conflict_graph import ConflictGraph
from .environment import (
    MultiCampaignStochasticEnv,
    NonStationaryMultiCampaignEnv,
    StochasticSingleCampaignEnv,
)


@dataclass
class EpisodeResult:
    utilities: np.ndarray       # per-round realized utility (0 after stopping)
    costs: np.ndarray           # per-round realized cost
    bid_indices: np.ndarray     # index of bid played (-1 after stopping)
    remaining_budget: np.ndarray
    wins: np.ndarray
    competing_bids: np.ndarray  # m_t (for diagnostics; not seen by the bidder)
    stopped_round: int          # round at which the budget was exhausted (or T)

    @property
    def total_utility(self) -> float:
        return float(self.utilities.sum())

    @property
    def total_cost(self) -> float:
        return float(self.costs.sum())


def run_episode(
    env: StochasticSingleCampaignEnv,
    bidder: Bidder,
    budget: float,
) -> EpisodeResult:
    horizon = env.horizon
    bids = env.bids
    max_bid = float(bids.max())

    utilities = np.zeros(horizon)
    costs = np.zeros(horizon)
    bid_indices = np.full(horizon, -1, dtype=np.int64)
    remaining_budget = np.zeros(horizon)
    wins = np.zeros(horizon, dtype=bool)
    competing_bids = np.full(horizon, np.nan)

    bidder.reset()
    remaining = float(budget)
    stopped_round = horizon

    for t in range(horizon):
        # Termination rule: cannot afford the largest possible cost.
        if remaining < max_bid:
            stopped_round = t
            remaining_budget[t:] = remaining
            break

        idx = bidder.select_bid_index()
        bid = float(bids[idx])
        outcome = env.step(bid)

        remaining -= outcome.cost
        bidder.update(idx, outcome.utility, outcome.cost)

        utilities[t] = outcome.utility
        costs[t] = outcome.cost
        bid_indices[t] = idx
        wins[t] = outcome.won
        competing_bids[t] = outcome.competing_bid
        remaining_budget[t] = remaining

    return EpisodeResult(
        utilities=utilities,
        costs=costs,
        bid_indices=bid_indices,
        remaining_budget=remaining_budget,
        wins=wins,
        competing_bids=competing_bids,
        stopped_round=stopped_round,
    )


# --------------------------------------------------------------------------- #
# Multi-campaign (Requirement 2) episode runner.
# --------------------------------------------------------------------------- #
@dataclass
class MultiEpisodeResult:
    utilities: np.ndarray          # per-round total realized utility
    costs: np.ndarray              # per-round total realized cost
    remaining_budget: np.ndarray
    n_active: np.ndarray           # per-round number of campaigns played
    bid_counts: np.ndarray         # (N, B) how often each (campaign, bid) was played
    stopped_round: int

    @property
    def total_utility(self) -> float:
        return float(self.utilities.sum())

    @property
    def total_cost(self) -> float:
        return float(self.costs.sum())


def run_multi_episode(
    env: MultiCampaignStochasticEnv,
    bidder: CombinatorialUCBBidder,
    budget: float,
) -> MultiEpisodeResult:
    """Run a Combinatorial-UCB bidder against the multi-campaign environment.

    Termination mirrors the single-campaign rule but accounts for several
    simultaneous campaigns: the agency stops once the remaining budget can no
    longer cover the worst-case round cost, i.e. ``(max independent set size) *
    max_bid``.  This guarantees the global budget is never exceeded.
    """
    horizon = env.horizon
    bids = env.bids
    max_bid = float(bids.max())
    cg = ConflictGraph(env.n_campaigns, env.scenario.edge_list())
    max_round_cost = cg.max_independent_set_size() * max_bid

    utilities = np.zeros(horizon)
    costs = np.zeros(horizon)
    remaining_budget = np.zeros(horizon)
    n_active = np.zeros(horizon, dtype=np.int64)
    bid_counts = np.zeros((env.n_campaigns, bids.size))

    bidder.reset()
    remaining = float(budget)
    stopped_round = horizon

    for t in range(horizon):
        if remaining < max_round_cost:
            stopped_round = t
            remaining_budget[t:] = remaining
            break

        actions = bidder.select_superarm()
        outcome = env.step(actions)

        remaining -= outcome.total_cost
        bidder.update(outcome.per_campaign)

        utilities[t] = outcome.total_utility
        costs[t] = outcome.total_cost
        remaining_budget[t] = remaining
        n_active[t] = len(actions)
        for i, bid_index in actions.items():
            bid_counts[i, bid_index] += 1

    return MultiEpisodeResult(
        utilities=utilities,
        costs=costs,
        remaining_budget=remaining_budget,
        n_active=n_active,
        bid_counts=bid_counts,
        stopped_round=stopped_round,
    )


# --------------------------------------------------------------------------- #
# Primal-dual (Requirement 3) episode runner -- full feedback.
# --------------------------------------------------------------------------- #
def run_primal_dual_episode(
    env: MultiCampaignStochasticEnv | NonStationaryMultiCampaignEnv,
    bidder: PrimalDualBidder,
    budget: float,
) -> MultiEpisodeResult:
    """Run the primal-dual bidder against a (stochastic or non-stationary) env.

    Identical budget bookkeeping / termination rule as :func:`run_multi_episode`,
    but the bidder is given **full feedback** each round: the entire competing-bid
    vector ``m`` (via :class:`~ola.environment.MultiOutcome.competing_bids`), as
    Requirement 3 allows.  This lets the Hedge primal regret minimizer score every
    (campaign, bid) pair, not only the played ones.
    """
    horizon = env.horizon
    bids = env.bids
    max_bid = float(bids.max())
    cg = ConflictGraph(env.n_campaigns, env.scenario.edge_list())
    max_round_cost = cg.max_independent_set_size() * max_bid

    utilities = np.zeros(horizon)
    costs = np.zeros(horizon)
    remaining_budget = np.zeros(horizon)
    n_active = np.zeros(horizon, dtype=np.int64)
    bid_counts = np.zeros((env.n_campaigns, bids.size))

    bidder.reset()
    remaining = float(budget)
    stopped_round = horizon

    for t in range(horizon):
        if remaining < max_round_cost:
            stopped_round = t
            remaining_budget[t:] = remaining
            break

        actions = bidder.select_superarm()
        outcome = env.step(actions)

        remaining -= outcome.total_cost
        bidder.update(outcome.per_campaign, outcome.competing_bids)

        utilities[t] = outcome.total_utility
        costs[t] = outcome.total_cost
        remaining_budget[t] = remaining
        n_active[t] = len(actions)
        for i, bid_index in actions.items():
            bid_counts[i, bid_index] += 1

    return MultiEpisodeResult(
        utilities=utilities,
        costs=costs,
        remaining_budget=remaining_budget,
        n_active=n_active,
        bid_counts=bid_counts,
        stopped_round=stopped_round,
    )
