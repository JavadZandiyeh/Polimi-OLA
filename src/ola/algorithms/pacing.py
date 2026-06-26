"""Single-campaign primal-dual pacing -- the special case of Requirement 3.

The best-of-both-worlds primal-dual strategy (slide 16 of
``8-GeneralAuctions.pdf``) for a *single* campaign is:

    initialize lambda <- 0, rho <- B / T
    each round t:
        gamma_t <- R(t)                              # primal regret minimizer
        bid b_t ~ gamma_t                            # over the Lagrangian reward
        observe f_t(b_t), c_t(b_t)                    #   L(b, lambda, f, c)
        lambda <- proj_[0, 1/rho]( lambda - eta (rho - c_t(b_t)) )   # dual OGD

with ``R`` a regret minimizer (Hedge with the full feedback on ``m_t`` granted by
Requirement 3) for the Lagrangian reward
``L(b, lambda, f, c) = f(b) - lambda (c(b) - rho)``.

This single-campaign recipe is exactly the one-block special case of the
multi-campaign best-of-both-worlds bidder implemented in
:class:`ola.algorithms.primal_dual.PrimalDualBidder` (run it with ``N = 1`` and
no conflict edges).  This module is kept as a focused, single-campaign reference;
use :class:`~ola.algorithms.primal_dual.PrimalDualBidder` for the actual
Requirement 3 experiments.
"""

from __future__ import annotations

import numpy as np

from .primal_dual import PrimalDualBidder


class PacingBidder(PrimalDualBidder):
    """Single-campaign primal-dual pacing bidder (one block, no conflicts).

    Thin convenience wrapper around :class:`PrimalDualBidder` for the
    single-campaign setting: ``L(b, lambda) = f(b) - lambda c(b)`` with a Hedge
    primal regret minimizer over the bid set and dual OGD on ``lambda``.
    """

    def __init__(
        self,
        bids: np.ndarray,
        valuation: float,
        horizon: int,
        rho: float,
        rng: np.random.Generator,
        **kwargs,
    ):
        super().__init__(
            valuations=np.asarray([valuation], dtype=float),
            bids=bids,
            horizon=horizon,
            rho=rho,
            conflict_edges=[],
            rng=rng,
            **kwargs,
        )

    def select_bid_index(self) -> int:
        """Single-campaign convenience: return the played bid index (or 0=skip)."""
        actions = self.select_superarm()
        return int(actions.get(0, 0))
