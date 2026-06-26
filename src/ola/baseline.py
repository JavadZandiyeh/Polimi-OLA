"""Clairvoyant baselines and the per-round bidding LP.

These quantities use the *true* distribution ``D`` and define the benchmarks
against which the online algorithms' regret is measured.

* Algorithm A (UCB1, no budget) is compared to the **best fixed bid in
  hindsight**, ``max_b E[f(b)]``.
* Algorithm B (budgeted UCB) is compared to the **constrained clairvoyant**
  ``OPT = sup_gamma E[f(gamma)]`` s.t. ``E[c(gamma)] <= rho`` -- the best
  (possibly randomized) fixed bidding distribution that respects the per-round
  budget in expectation (slide 10 of ``8-GeneralAuctions.pdf``).

The same LP that defines ``OPT`` is reused online by :mod:`ola.algorithms.ucb_budget`
with optimistic (UCB/LCB) estimates instead of the true means, so it lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .distributions import CompetingBidDistribution

if TYPE_CHECKING:
    from .environment import MultiCampaignScenario


def expected_utility_per_bid(
    valuation: float, bids: np.ndarray, dist: CompetingBidDistribution
) -> np.ndarray:
    """``E[f(b)] = (v - b) * P(m <= b)`` for every bid ``b``."""
    bids = np.asarray(bids, dtype=float)
    return (valuation - bids) * dist.cdf(bids)


def expected_cost_per_bid(bids: np.ndarray, dist: CompetingBidDistribution) -> np.ndarray:
    """``E[c(b)] = b * P(m <= b)`` for every bid ``b``."""
    bids = np.asarray(bids, dtype=float)
    return bids * dist.cdf(bids)


@dataclass
class FixedBidBaseline:
    bid_index: int
    bid: float
    expected_utility: float


def best_fixed_bid(
    valuation: float, bids: np.ndarray, dist: CompetingBidDistribution
) -> FixedBidBaseline:
    """Best fixed bid in hindsight (ignores the budget) -- baseline for UCB1."""
    f = expected_utility_per_bid(valuation, bids, dist)
    idx = int(np.argmax(f))
    return FixedBidBaseline(bid_index=idx, bid=float(bids[idx]), expected_utility=float(f[idx]))


@dataclass
class BiddingLPSolution:
    value: float            # objective: expected per-round utility
    gamma: np.ndarray       # distribution over bids
    expected_cost: float    # expected per-round cost of gamma


def solve_bidding_lp(
    f_values: np.ndarray, c_values: np.ndarray, rho: float
) -> BiddingLPSolution:
    """Solve ``max_gamma f.gamma s.t. c.gamma <= rho, gamma in simplex`` exactly.

    This LP has only two structural constraints (the simplex sum and the single
    budget constraint), so an optimal vertex is supported on **at most two**
    bids.  We therefore solve it in closed form:

    * if the highest-utility bid is already affordable (``c <= rho``), play it
      purely (mixing cannot beat ``max f``);
    * otherwise the optimum mixes two bids that straddle ``rho`` so the expected
      cost equals ``rho``; enumerate all such pairs (O(n^2), n is tiny) and keep
      the best.

    Geometrically this is the highest point of the convex hull of the points
    ``(c_b, f_b)`` lying in the half-plane ``c <= rho``.  It is always feasible
    because a zero-cost bid (or the cheapest bid) satisfies the constraint.
    ``solve_bidding_lp_linprog`` provides an independent ``scipy`` cross-check.
    """
    f_values = np.asarray(f_values, dtype=float)
    c_values = np.asarray(c_values, dtype=float)
    n = f_values.size

    # Case 1: budget inactive -- pure highest-utility bid if affordable.
    k_star = int(np.argmax(f_values))
    if c_values[k_star] <= rho:
        gamma = np.zeros(n)
        gamma[k_star] = 1.0
        return BiddingLPSolution(float(f_values[k_star]), gamma, float(c_values[k_star]))

    # Case 2: enumerate the best feasible pure bid and the best straddling pair.
    best_val = -np.inf
    best_pair: tuple[int, float, int] | None = None  # (i, alpha, j)

    affordable = np.where(c_values <= rho)[0]
    if affordable.size > 0:
        i_pure = int(affordable[np.argmax(f_values[affordable])])
        best_val = float(f_values[i_pure])
        best_pair = (i_pure, 1.0, i_pure)

    cheap = np.where(c_values < rho)[0]
    expensive = np.where(c_values > rho)[0]
    for i in cheap:
        ci, fi = c_values[i], f_values[i]
        for j in expensive:
            cj, fj = c_values[j], f_values[j]
            alpha = (rho - cj) / (ci - cj)  # weight on the cheap bid i
            if 0.0 <= alpha <= 1.0:
                val = alpha * fi + (1.0 - alpha) * fj
                if val > best_val:
                    best_val = val
                    best_pair = (int(i), float(alpha), int(j))

    gamma = np.zeros(n)
    if best_pair is None:
        # Constraint infeasible for every bid: fall back to the cheapest bid.
        gamma[int(np.argmin(c_values))] = 1.0
    else:
        i, alpha, j = best_pair
        gamma[i] += alpha
        gamma[j] += 1.0 - alpha

    return BiddingLPSolution(
        value=float(f_values @ gamma),
        gamma=gamma,
        expected_cost=float(c_values @ gamma),
    )


def solve_bidding_lp_linprog(
    f_values: np.ndarray, c_values: np.ndarray, rho: float
) -> BiddingLPSolution:
    """Reference ``scipy.linprog`` solver for :func:`solve_bidding_lp` (testing)."""
    from scipy.optimize import linprog

    f_values = np.asarray(f_values, dtype=float)
    c_values = np.asarray(c_values, dtype=float)
    n = f_values.size
    res = linprog(
        c=-f_values,
        A_ub=c_values.reshape(1, n),
        b_ub=np.array([rho], dtype=float),
        A_eq=np.ones((1, n)),
        b_eq=np.array([1.0]),
        bounds=[(0.0, 1.0)] * n,
        method="highs",
    )
    gamma = np.clip(res.x, 0.0, None)
    s = gamma.sum()
    gamma = gamma / s if s > 0 else gamma
    return BiddingLPSolution(
        value=float(f_values @ gamma),
        gamma=gamma,
        expected_cost=float(c_values @ gamma),
    )


def constrained_opt(
    valuation: float,
    bids: np.ndarray,
    dist: CompetingBidDistribution,
    rho: float,
) -> BiddingLPSolution:
    """Per-round constrained clairvoyant ``OPT`` -- baseline for budgeted UCB."""
    f = expected_utility_per_bid(valuation, bids, dist)
    c = expected_cost_per_bid(bids, dist)
    return solve_bidding_lp(f, c, rho)


# --------------------------------------------------------------------------- #
# Multi-campaign (Requirement 2) -- combinatorial LP oracle and baseline.
# --------------------------------------------------------------------------- #
def expected_utility_cost_matrices(
    valuations: np.ndarray,
    bids: np.ndarray,
    marginals: Sequence[CompetingBidDistribution],
) -> tuple[np.ndarray, np.ndarray]:
    """Per ``(campaign, bid)`` expected utility ``f[i,b]`` and cost ``c[i,b]``.

    Each entry uses only the campaign's marginal competing-bid distribution, so
    correlation across campaigns does not affect these expectations.
    """
    bids = np.asarray(bids, dtype=float)
    n = len(valuations)
    f = np.zeros((n, bids.size))
    c = np.zeros((n, bids.size))
    for i in range(n):
        cdf = marginals[i].cdf(bids)
        f[i] = (valuations[i] - bids) * cdf
        c[i] = bids * cdf
    return f, c


@dataclass
class MultiBiddingLPSolution:
    value: float            # expected per-round total utility
    x: np.ndarray           # (N, B) fractional allocation (x[i,b] = P(bid b in i))
    expected_cost: float    # expected per-round total cost


def solve_multi_campaign_lp(
    f: np.ndarray,
    c: np.ndarray,
    rho: float,
    conflict_edges: Sequence[tuple[int, int]] = (),
) -> MultiBiddingLPSolution:
    """Per-round combinatorial bidding LP -- the Combinatorial-UCB oracle.

        max_x   sum_{i,b} f[i,b] x[i,b]
        s.t.    sum_b x[i,b] <= 1                         (each campaign once)
                sum_{i,b} c[i,b] x[i,b] <= rho            (global per-round budget)
                sum_b x[i,b] + sum_b x[j,b] <= 1          (conflict edge (i,j))
                x >= 0

    ``x[i,b]`` is the probability of bidding ``b`` in campaign ``i`` this round;
    the leftover mass ``1 - sum_b x[i,b]`` is the probability of skipping the
    campaign.  For a conflict graph that is a matching (disjoint edges, as in the
    project) the edge constraints exactly describe the feasible (independent-set)
    region, so the LP is the right relaxation and admits exact conflict-aware
    rounding (see the sampler in :mod:`ola.algorithms.combinatorial_ucb`).
    The course assumes access to such an optimization oracle (Hungarian / LP).
    """
    from scipy.optimize import linprog

    f = np.asarray(f, dtype=float)
    c = np.asarray(c, dtype=float)
    n, b = f.shape
    nvar = n * b

    rows = [c.reshape(nvar)]
    rhs = [float(rho)]
    for i in range(n):
        row = np.zeros(nvar)
        row[i * b:(i + 1) * b] = 1.0
        rows.append(row)
        rhs.append(1.0)
    for (i, j) in conflict_edges:
        row = np.zeros(nvar)
        row[i * b:(i + 1) * b] = 1.0
        row[j * b:(j + 1) * b] = 1.0
        rows.append(row)
        rhs.append(1.0)

    res = linprog(
        c=-f.reshape(nvar),
        A_ub=np.vstack(rows),
        b_ub=np.array(rhs, dtype=float),
        bounds=[(0.0, 1.0)] * nvar,
        method="highs",
    )
    if not res.success or res.x is None:
        # Infeasible/failure: skip all campaigns (zero utility, zero cost).
        x = np.zeros((n, b))
        return MultiBiddingLPSolution(0.0, x, 0.0)

    x = np.clip(res.x, 0.0, None).reshape(n, b)
    return MultiBiddingLPSolution(float((f * x).sum()), x, float((c * x).sum()))


def multi_campaign_opt(
    scenario: "MultiCampaignScenario", rho: float | None = None
) -> MultiBiddingLPSolution:
    """Clairvoyant per-round ``OPT`` for the multi-campaign instance (true means)."""
    marginals = scenario.make_marginals()
    f, c = expected_utility_cost_matrices(
        scenario.valuation_array(), scenario.bid_array(), marginals
    )
    rho_value = scenario.rho if rho is None else rho
    return solve_multi_campaign_lp(f, c, rho_value, scenario.edge_list())


# --------------------------------------------------------------------------- #
# Best-of-both-worlds (Requirement 3) -- hindsight benchmark on a realized
# (possibly non-stationary / adversarial) sequence of competing bids.
# --------------------------------------------------------------------------- #
def empirical_utility_cost_matrices(
    valuations: np.ndarray,
    bids: np.ndarray,
    competing_bids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per ``(campaign, bid)`` *empirical* mean utility / cost over a realized run.

    ``competing_bids`` is the ``(T, N)`` sequence of highest competing bids that
    actually occurred (``competing_bids[t, i] = m_{i,t}``).  Because the agency
    has *full feedback* in Requirement 3 (it observes every campaign's ``m_i``),
    these averages are well defined for **every** ``(campaign, bid)`` pair, not
    only for the bids/campaigns that were played:

        f_bar[i, b] = (1/T) sum_t (v_i - b) * 1[b >= m_{i,t}]
        c_bar[i, b] = (1/T) sum_t  b        * 1[b >= m_{i,t}]
    """
    valuations = np.asarray(valuations, dtype=float)
    bids = np.asarray(bids, dtype=float)
    m = np.asarray(competing_bids, dtype=float)
    if m.ndim == 1:
        m = m[:, None]
    # win[t, i, b] = 1[bids[b] >= m[t, i]]
    win = (bids[None, None, :] >= m[:, :, None]).astype(float)
    f = ((valuations[None, :, None] - bids[None, None, :]) * win).mean(axis=0)
    c = (bids[None, None, :] * win).mean(axis=0)
    return f, c


def hindsight_multi_campaign_opt(
    valuations: np.ndarray,
    bids: np.ndarray,
    competing_bids: np.ndarray,
    rho: float,
    conflict_edges: Sequence[tuple[int, int]] = (),
) -> MultiBiddingLPSolution:
    """Best *fixed* feasible bidding distribution in hindsight on a realized run.

    This is the best-of-both-worlds benchmark for the primal-dual bidder: the
    single (per-campaign, possibly randomized) bidding distribution that, replayed
    against the *actual* sequence of competing bids, maximizes total utility while
    respecting the per-round budget ``rho`` (and the conflict graph).  It is the
    same combinatorial LP as :func:`solve_multi_campaign_lp`, fed the empirical
    utility/cost matrices of the realized sequence.

    * In a **stochastic** environment the empirical means concentrate on the true
      means, so this benchmark coincides (in the limit) with the req2 clairvoyant
      :func:`multi_campaign_opt`.
    * In a **non-stationary / adversarial** environment it is the strongest
      benchmark for which sublinear regret is information-theoretically possible
      (the best fixed action in hindsight), which is exactly what a
      best-of-both-worlds guarantee targets.
    """
    f, c = empirical_utility_cost_matrices(valuations, bids, competing_bids)
    return solve_multi_campaign_lp(f, c, rho, conflict_edges)


@dataclass
class PiecewiseDynamicOpt:
    total_value: float          # sum over intervals of len * per-round OPT
    total_cost: float           # sum over intervals of len * per-round cost
    n_intervals: int
    interval_values: list[float]  # per-round constrained OPT of each interval


def piecewise_dynamic_opt(scenario) -> PiecewiseDynamicOpt:
    """Per-interval clairvoyant optimum of a piecewise-stationary scenario.

    This is the *dynamic* (tracking) benchmark for Requirement 4: a policy that
    knows the interval structure and plays each interval's own constrained
    ``OPT`` (true means, per-round budget ``rho``).  It is the yardstick the
    sliding-window and change-detection learners aim to track; the sum is

        sum_p (interval length_p) * OPT(interval_p) .

    ``scenario`` must be a :class:`~ola.environment.NonStationaryMultiCampaignScenario`
    with an ``"abrupt"`` (piecewise-stationary) schedule.
    """
    schedule = scenario.schedule
    if schedule.get("kind") != "abrupt":
        raise ValueError("piecewise_dynamic_opt requires an 'abrupt' schedule")

    valuations = scenario.valuation_array()
    bids = scenario.bid_array()
    horizon = int(scenario.horizon)
    rho = scenario.rho
    edges = scenario.edge_list()
    phase_len = int(schedule["phase_length"])
    per_campaign = schedule["campaign_phase_dists"]
    n_configs = len(per_campaign[0])

    total_value = 0.0
    total_cost = 0.0
    interval_values: list[float] = []
    for start in range(0, horizon, phase_len):
        length = min(phase_len, horizon - start)
        phase_idx = (start // phase_len) % n_configs
        marginals = [
            CompetingBidDistribution.from_config(per_campaign[i][phase_idx])
            for i in range(len(valuations))
        ]
        f, c = expected_utility_cost_matrices(valuations, bids, marginals)
        sol = solve_multi_campaign_lp(f, c, rho, edges)
        total_value += length * sol.value
        total_cost += length * sol.expected_cost
        interval_values.append(float(sol.value))

    return PiecewiseDynamicOpt(
        total_value=float(total_value),
        total_cost=float(total_cost),
        n_intervals=len(interval_values),
        interval_values=interval_values,
    )
