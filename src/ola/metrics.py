"""Regret and budget bookkeeping.

The benchmarks are *per-round* expected values (the best fixed bid for
Algorithm A, the constrained ``OPT`` for Algorithm B).  The cumulative benchmark
after ``t`` rounds is ``t * per_round_value`` and the (pseudo-)regret is

    regret_t = t * per_round_value - sum_{s<=t} reward_s.

Because a budget-depleted run earns zero in its remaining rounds, while the
benchmark keeps accruing, an algorithm that wastes its budget early shows up as
*linear* regret -- which is exactly the contrast we want between Algorithm A and
Algorithm B.
"""

from __future__ import annotations

import numpy as np


def cumulative(x: np.ndarray) -> np.ndarray:
    return np.cumsum(np.asarray(x, dtype=float))


def regret_curve(rewards: np.ndarray, per_round_baseline: float) -> np.ndarray:
    """Cumulative pseudo-regret against a constant per-round benchmark."""
    rewards = np.asarray(rewards, dtype=float)
    horizon = rewards.size
    benchmark = per_round_baseline * np.arange(1, horizon + 1)
    return benchmark - np.cumsum(rewards)


def average_curves(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error across runs (rows = runs, cols = rounds)."""
    stacked = np.vstack(curves)
    mean = stacked.mean(axis=0)
    std_err = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0]) if stacked.shape[0] > 1 else np.zeros_like(mean)
    return mean, std_err
