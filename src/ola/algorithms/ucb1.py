"""Algorithm A: UCB1 bidding, ignoring the budget constraint.

Each discrete bid ``b in B`` is treated as an arm whose stochastic reward is the
first-price utility ``f_t(b) = (v - b) 1[b >= m_t] in [0, v]``.  We run the
classic UCB1 rule (Auer et al., 2002): pull every arm once, then play

    argmax_b  mean(b) + v * sqrt( 2 ln t / N(b) ).

The ``v`` factor rescales the standard ``[0, 1]`` confidence radius to the
actual reward range ``[0, v]`` (the maximum possible utility is ``v``, attained
by a winning zero-cost bid).

The budget is **not** used when choosing bids: the budget only matters because
the simulation stops the bidder once it can no longer afford a bid.  This makes
Algorithm A a useful contrast to the budget-aware Algorithm B.
"""

from __future__ import annotations

import numpy as np

from .base import Bidder


class UCB1Bidder(Bidder):
    def __init__(self, bids: np.ndarray, valuation: float, horizon: int):
        super().__init__(bids, valuation)
        self.horizon = int(horizon)
        # Reward range: utility lies in [0, v], so the largest possible reward
        # is `v - min(bid)`. Using `v` is a valid (tight when min bid is 0) bound.
        self.reward_range = max(self.valuation - float(self.bids.min()), 1e-12)
        self.reset()

    def reset(self) -> None:
        self.counts = np.zeros(self.n_bids, dtype=np.int64)
        self.sum_utility = np.zeros(self.n_bids, dtype=float)
        self.t = 0

    def select_bid_index(self) -> int:
        # Initialization phase: play each arm once.
        unplayed = np.where(self.counts == 0)[0]
        if unplayed.size > 0:
            return int(unplayed[0])

        means = self.sum_utility / self.counts
        bonus = self.reward_range * np.sqrt(2.0 * np.log(max(self.t, 1)) / self.counts)
        return int(np.argmax(means + bonus))

    def update(self, bid_index: int, utility: float, cost: float) -> None:
        # `cost` is intentionally ignored: this algorithm ignores the budget.
        self.counts[bid_index] += 1
        self.sum_utility[bid_index] += utility
        self.t += 1
