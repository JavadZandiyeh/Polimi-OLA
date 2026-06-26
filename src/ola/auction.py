"""First-price single-slot auction mechanics.

At each round the advertiser bids ``b`` and the highest competing bid is ``m``.
The advertiser wins iff ``b >= m``.  Under the *first-price* rule the winner
pays exactly its own bid, so:

    win   = 1[b >= m]
    utility f(b) = (v - b) * win
    cost    c(b) = b       * win

All helpers are vectorized: ``b`` and ``m`` may be scalars or NumPy arrays that
broadcast against each other.
"""

from __future__ import annotations

import numpy as np


def wins(b: np.ndarray | float, m: np.ndarray | float) -> np.ndarray:
    """Return the win indicator ``1[b >= m]``."""
    return (np.asarray(b, dtype=float) >= np.asarray(m, dtype=float)).astype(float)


def utility(v: float, b: np.ndarray | float, m: np.ndarray | float) -> np.ndarray:
    """First-price utility ``(v - b) * 1[b >= m]``."""
    b = np.asarray(b, dtype=float)
    return (v - b) * wins(b, m)


def cost(b: np.ndarray | float, m: np.ndarray | float) -> np.ndarray:
    """First-price cost ``b * 1[b >= m]``."""
    b = np.asarray(b, dtype=float)
    return b * wins(b, m)
