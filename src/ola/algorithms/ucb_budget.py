"""Algorithm B: UCB-bidding with a budget constraint.

This is the "UCB-like approach" for bandits-with-knapsacks bidding
(Agrawal & Devanur, 2014; slide 18 of ``8-GeneralAuctions.pdf``).  It is the
budget-aware extension of UCB1.

For every bid ``b`` we keep an **optimistic** estimate of the utility and a
**pessimistic** estimate of the cost:

    f_UCB(b) = mean_f(b) + sqrt( 2 ln T / N(b) )   (clipped to [0, v - b])
    c_LCB(b) = mean_c(b) - sqrt( 2 ln T / N(b) )   (clipped to [0, b])

and each round we play the distribution over bids solving

    gamma_t = argmax_gamma  sum_b gamma(b) f_UCB(b)
              s.t.          sum_b gamma(b) c_LCB(b) <= rho        (rho = B / T)
                            gamma in the simplex,

then sample ``b_t ~ gamma_t``.  Optimism on utility and pessimism on cost makes
the algorithm explore while keeping the realized spend close to the per-round
budget ``rho``.  Unseen arms get ``f_UCB = v - b`` and ``c_LCB = 0`` so the LP
is driven to try them.

Budget accounting and the "stop when you cannot afford a bid" rule are handled
by :mod:`ola.simulation`; the multiplier ``rho`` used in the LP is the fixed
per-round budget ``B / T``.
"""

from __future__ import annotations

import numpy as np

from ..baseline import solve_bidding_lp
from .base import Bidder


class BudgetedUCBBidder(Bidder):
    def __init__(
        self,
        bids: np.ndarray,
        valuation: float,
        horizon: int,
        rho: float,
        rng: np.random.Generator,
    ):
        super().__init__(bids, valuation)
        self.horizon = int(horizon)
        self.rho = float(rho)
        self.rng = rng
        # Per-bid hard bounds: utility in [0, v - b], cost in [0, b].
        self.max_utility = np.maximum(self.valuation - self.bids, 0.0)
        self.max_cost = self.bids.copy()
        self.reset()

    def reset(self) -> None:
        self.counts = np.zeros(self.n_bids, dtype=np.int64)
        self.sum_utility = np.zeros(self.n_bids, dtype=float)
        self.sum_cost = np.zeros(self.n_bids, dtype=float)
        self.t = 0
        self.last_gamma = np.full(self.n_bids, 1.0 / self.n_bids)

    def _confidence_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        log_term = 2.0 * np.log(max(self.horizon, 2))
        seen = self.counts > 0
        n = np.where(seen, self.counts, 1)  # avoid divide-by-zero for unseen

        mean_f = np.where(seen, self.sum_utility / n, 0.0)
        mean_c = np.where(seen, self.sum_cost / n, 0.0)
        # Hoeffding radius scaled by each quantity's range: utility lies in
        # [0, v - b] and cost in [0, b], so a bid's bounds tighten at the correct
        # rate (e.g. a zero bid has zero cost uncertainty). This pacing is what
        # keeps the realized spend close to rho.
        base = np.sqrt(log_term / n)
        f_ucb = np.clip(mean_f + self.max_utility * base, 0.0, self.max_utility)
        c_lcb = np.clip(mean_c - self.max_cost * base, 0.0, self.max_cost)

        # Unseen arms: fully optimistic (max utility, zero cost) to force trial.
        unseen = ~seen
        f_ucb[unseen] = self.max_utility[unseen]
        c_lcb[unseen] = 0.0
        return f_ucb, c_lcb

    def select_bid_index(self) -> int:
        f_ucb, c_lcb = self._confidence_bounds()
        solution = solve_bidding_lp(f_ucb, c_lcb, self.rho)
        gamma = solution.gamma
        self.last_gamma = gamma
        return int(self.rng.choice(self.n_bids, p=gamma))

    def update(self, bid_index: int, utility: float, cost: float) -> None:
        self.counts[bid_index] += 1
        self.sum_utility[bid_index] += utility
        self.sum_cost[bid_index] += cost
        self.t += 1
