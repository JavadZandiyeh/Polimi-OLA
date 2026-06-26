"""Bidding algorithms.

* :class:`~ola.algorithms.ucb1.UCB1Bidder` -- Algorithm A (Requirement 1):
  UCB1 over the discrete bid set, ignoring the budget constraint.
* :class:`~ola.algorithms.ucb_budget.BudgetedUCBBidder` -- Algorithm B
  (Requirement 1): UCB-bidding with a per-round LP that enforces the budget.
* :class:`~ola.algorithms.combinatorial_ucb.CombinatorialUCBBidder` --
  Requirement 2: Combinatorial-UCB with a budget constraint over many campaigns.
* :class:`~ola.algorithms.primal_dual.PrimalDualBidder` -- Requirement 3:
  best-of-both-worlds primal-dual bidding with a Hedge primal regret minimizer
  (handles stochastic *and* highly non-stationary environments, full feedback).
* :class:`~ola.algorithms.nonstationary_ucb.SlidingWindowCombinatorialUCBBidder`
  and
  :class:`~ola.algorithms.nonstationary_ucb.ChangeDetectionCombinatorialUCBBidder`
  -- Requirement 4: Combinatorial-UCB with a sliding window / a CUSUM change
  detector for slightly (piecewise-stationary) non-stationary environments.
* :mod:`~ola.algorithms.pacing` -- single-campaign primal-dual pacing notes.
"""

from .base import Bidder
from .combinatorial_ucb import CombinatorialUCBBidder
from .nonstationary_ucb import (
    ChangeDetectionCombinatorialUCBBidder,
    SlidingWindowCombinatorialUCBBidder,
)
from .primal_dual import PrimalDualBidder
from .ucb1 import UCB1Bidder
from .ucb_budget import BudgetedUCBBidder

__all__ = [
    "Bidder",
    "UCB1Bidder",
    "BudgetedUCBBidder",
    "CombinatorialUCBBidder",
    "PrimalDualBidder",
    "SlidingWindowCombinatorialUCBBidder",
    "ChangeDetectionCombinatorialUCBBidder",
]
