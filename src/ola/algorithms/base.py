"""Common interface for bidding algorithms.

A bidder picks a bid (by index into the discrete bid set) each round, then
receives the bandit feedback of that bid only: its realized utility and cost.
The simulation loop (:mod:`ola.simulation`) owns the budget accounting, so
bidders that ignore the budget and bidders that pace it share the same API.
"""

from __future__ import annotations

import numpy as np


class Bidder:
    """Abstract base class for a single-campaign bidder."""

    def __init__(self, bids: np.ndarray, valuation: float):
        self.bids = np.asarray(bids, dtype=float)
        self.n_bids = self.bids.size
        self.valuation = float(valuation)

    def reset(self) -> None:
        """Reset internal state for a fresh run."""
        raise NotImplementedError

    def select_bid_index(self) -> int:
        """Return the index (into ``self.bids``) of the bid to play this round."""
        raise NotImplementedError

    def update(self, bid_index: int, utility: float, cost: float) -> None:
        """Incorporate the bandit feedback of the played bid."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__
