"""Requirement 2: Combinatorial-UCB with a budget constraint.

This extends the single-campaign "UCB-like" budgeted bidder (Requirement 1) to
``N`` simultaneous first-price campaigns, following the Combinatorial-UCB recipe
(Part 9 of the course): treat each ``(campaign, bid)`` pair as an arm, keep an
optimistic confidence bound on its mean, and each round play the best feasible
**superarm** via an optimization oracle.

* Arm ``(i, b)``: expected utility ``f_i(b)`` and cost ``c_i(b)``.
* Optimism: ``f_UCB(i,b) = mean_f + range * sqrt(2 log T / N)`` and
  ``c_LCB(i,b) = mean_c - range * sqrt(2 log T / N)`` (cost is treated
  pessimistically so the budget is respected while exploring).
* Oracle (slide 42, with the budget added): solve the per-round LP

      max_x sum f_UCB . x  s.t. per-campaign <=1, budget sum c_LCB . x <= rho,
                                 conflict edges,  x >= 0

  (see :func:`ola.baseline.solve_multi_campaign_lp`). The fractional solution
  ``x[i,b]`` is then rounded into an actual joint action with conflict-aware
  sampling: independent campaigns are sampled with a Bernoulli "play" probability
  ``sum_b x[i,b]``, while each conflict pair is rounded mutually exclusively so
  the realized action is always a feasible independent set.

Feedback is semi-bandit: only the played campaigns' arms are updated.  The budget
multiplier is the fixed per-round budget ``rho = B / T``; the actual budget
accounting / termination lives in :mod:`ola.simulation`.
"""

from __future__ import annotations

import numpy as np

from ..baseline import solve_multi_campaign_lp


class CombinatorialUCBBidder:
    def __init__(
        self,
        valuations: np.ndarray,
        bids: np.ndarray,
        horizon: int,
        rho: float,
        conflict_edges: list[tuple[int, int]],
        rng: np.random.Generator,
    ):
        self.valuations = np.asarray(valuations, dtype=float)
        self.bids = np.asarray(bids, dtype=float)
        self.n_campaigns = self.valuations.size
        self.n_bids = self.bids.size
        self.horizon = int(horizon)
        self.rho = float(rho)
        self.conflict_edges = [(int(i), int(j)) for i, j in conflict_edges]
        self.rng = rng

        # Per-arm hard ranges: utility in [0, v_i - b], cost in [0, b].
        self.max_utility = np.maximum(self.valuations[:, None] - self.bids[None, :], 0.0)
        self.max_cost = np.broadcast_to(self.bids[None, :], self.max_utility.shape).copy()

        # Campaigns appearing in a conflict edge vs. "free" campaigns.
        self._edge_nodes = {n for e in self.conflict_edges for n in e}
        self._free_nodes = [i for i in range(self.n_campaigns) if i not in self._edge_nodes]
        self.reset()

    def reset(self) -> None:
        shape = (self.n_campaigns, self.n_bids)
        self.counts = np.zeros(shape, dtype=np.int64)
        self.sum_utility = np.zeros(shape, dtype=float)
        self.sum_cost = np.zeros(shape, dtype=float)
        self.t = 0
        self.last_x = np.zeros(shape)

    def _confidence_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        log_term = 2.0 * np.log(max(self.horizon, 2))
        seen = self.counts > 0
        n = np.where(seen, self.counts, 1)
        mean_f = np.where(seen, self.sum_utility / n, 0.0)
        mean_c = np.where(seen, self.sum_cost / n, 0.0)
        base = np.sqrt(log_term / n)

        f_ucb = np.clip(mean_f + self.max_utility * base, 0.0, self.max_utility)
        c_lcb = np.clip(mean_c - self.max_cost * base, 0.0, self.max_cost)
        unseen = ~seen
        f_ucb[unseen] = self.max_utility[unseen]
        c_lcb[unseen] = 0.0
        return f_ucb, c_lcb

    def _sample_superarm(self, x: np.ndarray) -> dict[int, int]:
        """Round the fractional LP solution into a feasible joint action."""
        play_prob = x.sum(axis=1)  # P(campaign i is active)
        actions: dict[int, int] = {}

        def pick_bid(i: int) -> int:
            p = x[i] / play_prob[i]
            p = np.clip(p, 0.0, None)
            p /= p.sum()
            return int(self.rng.choice(self.n_bids, p=p))

        # Conflict pairs: mutually exclusive rounding (exact for a matching).
        for i, j in self.conflict_edges:
            ai, aj = play_prob[i], play_prob[j]
            r = self.rng.random()
            if r < ai:
                actions[i] = pick_bid(i)
            elif r < ai + aj:
                actions[j] = pick_bid(j)
            # else: neither campaign is played this round.

        # Free campaigns: independent Bernoulli rounding.
        for i in self._free_nodes:
            if play_prob[i] > 1e-12 and self.rng.random() < play_prob[i]:
                actions[i] = pick_bid(i)
        return actions

    def select_superarm(self) -> dict[int, int]:
        f_ucb, c_lcb = self._confidence_bounds()
        solution = solve_multi_campaign_lp(f_ucb, c_lcb, self.rho, self.conflict_edges)
        self.last_x = solution.x
        return self._sample_superarm(solution.x)

    def update(self, per_campaign: dict[int, tuple[int, bool, float, float]]) -> None:
        for i, (bid_index, _won, util, cost) in per_campaign.items():
            self.counts[i, bid_index] += 1
            self.sum_utility[i, bid_index] += util
            self.sum_cost[i, bid_index] += cost
        self.t += 1

    @property
    def name(self) -> str:
        return type(self).__name__
