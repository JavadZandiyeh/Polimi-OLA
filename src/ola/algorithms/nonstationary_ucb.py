"""Requirement 4: non-stationary extensions of Combinatorial-UCB.

For a *slightly* non-stationary (piecewise-stationary) multi-campaign
environment -- the horizon is split into intervals, each with its own fixed
distribution of highest competing bids -- the stochastic Combinatorial-UCB of
Requirement 2 fails: it shrinks its confidence bounds and "converges", so it
cannot react when an interval boundary changes the best bid.  Part 10 of the
course gives two standard fixes, here lifted to the combinatorial / budgeted
setting of Requirement 2:

* :class:`SlidingWindowCombinatorialUCBBidder` -- **passive** forgetting.  Every
  ``(campaign, bid)`` arm is estimated from only the last ``W`` rounds, so the
  confidence bounds never collapse and old (stale) samples are dropped
  (slides 37-42).  A good rule of thumb is ``W ~ sqrt(T)``, enlarged here because
  the combinatorial action spreads plays over many arms.

* :class:`ChangeDetectionCombinatorialUCBBidder` -- **active** detection.  A
  per-arm CUSUM change detector (slides 68-72) monitors each arm's utility; when
  a change is flagged the arm's statistics are reset (it becomes "fresh" and
  optimistic again), so the learner re-estimates the new distribution.  A small
  uniform exploration probability ``alpha`` keeps feeding samples to every arm so
  changes can actually be detected (slide 65).

Both reuse Requirement 2's per-round budget LP oracle and conflict-aware rounding
(:class:`~ola.algorithms.combinatorial_ucb.CombinatorialUCBBidder`); only the
sufficient statistics fed to the optimistic bounds change.  Feedback stays
semi-bandit (only played campaigns are observed), exactly as in Requirement 2.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .combinatorial_ucb import CombinatorialUCBBidder


class SlidingWindowCombinatorialUCBBidder(CombinatorialUCBBidder):
    """Combinatorial-UCB whose per-arm estimates use only the last ``W`` rounds."""

    def __init__(
        self,
        valuations: np.ndarray,
        bids: np.ndarray,
        horizon: int,
        rho: float,
        conflict_edges: list[tuple[int, int]],
        rng: np.random.Generator,
        window: int | None = None,
    ):
        # Window size. The textbook rule is W ~ sqrt(T); here it is enlarged
        # (~20 sqrt(T)) because the combinatorial, budget-constrained action
        # spreads plays over many (campaign, bid) arms, so a plain sqrt(T) window
        # leaves each arm with too few samples to estimate (and then permanently
        # over-explores under the budget).  The best W still depends on the
        # unknown change frequency (course remark), so it is configurable.
        self.window = int(window) if window is not None else int(20 * np.sqrt(max(horizon, 2)))
        super().__init__(valuations, bids, horizon, rho, conflict_edges, rng)

    def reset(self) -> None:
        super().reset()
        # Ring buffer of the last W rounds; each entry is that round's list of
        # (campaign, bid_index, utility, cost) observations (semi-bandit).
        self._history: deque[list[tuple[int, int, float, float]]] = deque()

    def update(self, per_campaign: dict[int, tuple[int, bool, float, float]]) -> None:
        entry: list[tuple[int, int, float, float]] = []
        for i, (bid_index, _won, util, cost) in per_campaign.items():
            self.counts[i, bid_index] += 1
            self.sum_utility[i, bid_index] += util
            self.sum_cost[i, bid_index] += cost
            entry.append((i, bid_index, util, cost))
        self._history.append(entry)
        self.t += 1

        # Evict observations that have slid out of the window.
        if len(self._history) > self.window:
            for i, bid_index, util, cost in self._history.popleft():
                self.counts[i, bid_index] -= 1
                self.sum_utility[i, bid_index] -= util
                self.sum_cost[i, bid_index] -= cost

    @property
    def name(self) -> str:
        return "SW-CombUCB"


class ChangeDetectionCombinatorialUCBBidder(CombinatorialUCBBidder):
    """Combinatorial-UCB with a per-arm CUSUM change detector and uniform exploration.

    CUSUM (per ``(campaign, bid)`` arm), monitoring the utility stream:

    * **Estimation phase** -- the first ``cusum_m`` samples after a (re)start set
      the reference mean ``mu_bar``;
    * **Detection phase** -- for each later sample ``x`` accumulate the positive
      and negative deviations
      ``g+ = max(0, g+ + (x - mu_bar - eps))`` and
      ``g- = max(0, g- + (-(x - mu_bar) - eps))``;
      a change is flagged (and the arm reset) when ``g+ > h`` or ``g- > h``.

    With probability ``alpha`` the round plays a uniformly random single arm
    instead of the optimistic LP action, guaranteeing every arm keeps being
    sampled so that changes can be detected.
    """

    def __init__(
        self,
        valuations: np.ndarray,
        bids: np.ndarray,
        horizon: int,
        rho: float,
        conflict_edges: list[tuple[int, int]],
        rng: np.random.Generator,
        cusum_m: int = 50,
        cusum_eps: float = 0.15,
        cusum_h: float = 5.0,
        alpha: float = 0.01,
    ):
        self.cusum_m = int(cusum_m)
        self.cusum_eps = float(cusum_eps)
        self.cusum_h = float(cusum_h)
        self.alpha = float(alpha)
        super().__init__(valuations, bids, horizon, rho, conflict_edges, rng)

    def reset(self) -> None:
        super().reset()
        shape = (self.n_campaigns, self.n_bids)
        self._cs_count = np.zeros(shape, dtype=np.int64)  # samples since (re)start
        self._cs_mean = np.zeros(shape, dtype=float)      # estimation-phase mean
        self._g_plus = np.zeros(shape, dtype=float)
        self._g_minus = np.zeros(shape, dtype=float)
        self.n_detections = 0

    # ------------------------------------------------------------------ #
    def _reset_arm(self, i: int, b: int) -> None:
        self.counts[i, b] = 0
        self.sum_utility[i, b] = 0.0
        self.sum_cost[i, b] = 0.0
        self._cs_count[i, b] = 0
        self._cs_mean[i, b] = 0.0
        self._g_plus[i, b] = 0.0
        self._g_minus[i, b] = 0.0

    def _cusum_flags_change(self, i: int, b: int, reward: float) -> bool:
        """Feed ``reward`` to arm ``(i, b)``'s CUSUM; return True if it changed."""
        if self._cs_count[i, b] < self.cusum_m:
            # Estimation phase: running mean of the first cusum_m samples.
            n = self._cs_count[i, b]
            self._cs_mean[i, b] = (self._cs_mean[i, b] * n + reward) / (n + 1)
            self._cs_count[i, b] = n + 1
            return False
        # Detection phase.
        dev = reward - self._cs_mean[i, b]
        self._g_plus[i, b] = max(0.0, self._g_plus[i, b] + dev - self.cusum_eps)
        self._g_minus[i, b] = max(0.0, self._g_minus[i, b] - dev - self.cusum_eps)
        return self._g_plus[i, b] > self.cusum_h or self._g_minus[i, b] > self.cusum_h

    # ------------------------------------------------------------------ #
    def _random_superarm(self) -> dict[int, int]:
        """A uniformly random single-arm action (always conflict-feasible)."""
        i = int(self.rng.integers(self.n_campaigns))
        b = int(self.rng.integers(self.n_bids))
        return {i: b}

    def select_superarm(self) -> dict[int, int]:
        if self.rng.random() < self.alpha:
            return self._random_superarm()
        return super().select_superarm()

    def update(self, per_campaign: dict[int, tuple[int, bool, float, float]]) -> None:
        for i, (bid_index, _won, util, cost) in per_campaign.items():
            if self._cusum_flags_change(i, bid_index, util):
                # Change detected: drop this arm's history and re-seed with the
                # current observation (so it restarts its estimation phase).
                self._reset_arm(i, bid_index)
                self.n_detections += 1
            self.counts[i, bid_index] += 1
            self.sum_utility[i, bid_index] += util
            self.sum_cost[i, bid_index] += cost
        self.t += 1

    @property
    def name(self) -> str:
        return "CD-CombUCB"
