"""Synthesize and persist the datasets for the bidding project.

The "dataset" of a stochastic environment is the distribution of the highest
competing bid ``m_t`` together with the instance parameters (valuation, discrete
bid set, horizon, budget).  This script writes, for each scenario:

* a JSON config in ``data/scenarios/`` (the canonical, reproducible description);
* a CSV in ``data/samples/`` containing one reproducible realization of the
  ``m_t`` sequence, for inspection / sanity checks.

For every base distribution we emit two budget regimes:

* ``*_binding`` -- the per-round budget ``rho`` is set to half the spend of the
  unconstrained best bid, so the budget constraint is *active* (this is where
  Algorithm A and Algorithm B differ);
* ``*_slack``   -- ``rho = max_bid`` so the budget never binds (Algorithm B then
  behaves like an unconstrained utility maximizer, matching Algorithm A's
  benchmark).

Multi-campaign + conflict-graph scenarios are also written (data only) so the
later requirements have ready-made inputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ola.baseline import (  # noqa: E402
    best_fixed_bid,
    expected_cost_per_bid,
    hindsight_multi_campaign_opt,
    multi_campaign_opt,
)
from ola.conflict_graph import ConflictGraph  # noqa: E402
from ola.distributions import CompetingBidDistribution  # noqa: E402
from ola.environment import (  # noqa: E402
    MultiCampaignScenario,
    NonStationaryMultiCampaignScenario,
    Scenario,
)

SCENARIO_DIR = ROOT / "data" / "scenarios"
SAMPLE_DIR = ROOT / "data" / "samples"

HORIZON = 10_000
BID_STEP = 0.05
BIDS = [round(b, 2) for b in np.arange(0.0, 1.0 + 1e-9, BID_STEP)]

# (name, valuation, description, distribution-config)
SINGLE_DISTRIBUTIONS = [
    (
        "uniform",
        0.8,
        "Highest competing bid uniform on [0, 1]: neutral baseline.",
        {"type": "uniform", "low": 0.0, "high": 1.0},
    ),
    (
        "low_competition",
        0.8,
        "Weak competitors (truncated normal, mean 0.2): many cheap wins.",
        {"type": "truncnorm", "mean": 0.2, "std": 0.1, "low": 0.0, "high": 1.0},
    ),
    (
        "high_competition",
        0.9,
        "Strong competitors (truncated normal, mean 0.7): wins are expensive.",
        {"type": "truncnorm", "mean": 0.7, "std": 0.1, "low": 0.0, "high": 1.0},
    ),
    (
        "beta_skewed",
        0.8,
        "Beta(2, 5): mostly low competing bids with an upper tail.",
        {"type": "beta", "a": 2.0, "b": 5.0},
    ),
    (
        "bimodal",
        0.85,
        "Mixture of a cheap mode (0.2) and an expensive mode (0.75): stresses pacing.",
        {
            "type": "bimodal",
            "weight_low": 0.6,
            "mean_low": 0.2,
            "std_low": 0.07,
            "mean_high": 0.75,
            "std_high": 0.07,
        },
    ),
]


def _greedy_expected_cost(valuation: float, dist_config: dict) -> float:
    bids = np.asarray(BIDS, dtype=float)
    dist = CompetingBidDistribution.from_config(dist_config)
    best = best_fixed_bid(valuation, bids, dist)
    return float(expected_cost_per_bid(bids, dist)[best.bid_index])


def build_single_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    max_bid = max(BIDS)
    for seed, (name, valuation, desc, dist_config) in enumerate(SINGLE_DISTRIBUTIONS):
        greedy_cost = _greedy_expected_cost(valuation, dist_config)

        # Budget-binding regime: per-round budget = half the greedy spend.
        rho_binding = max(0.5 * greedy_cost, BID_STEP)
        budget_binding = round(rho_binding * HORIZON, 2)
        scenarios.append(
            Scenario(
                name=f"{name}_binding",
                description=f"{desc} Budget-binding regime (rho approx {rho_binding:.3f}).",
                valuation=valuation,
                bids=BIDS,
                horizon=HORIZON,
                budget=budget_binding,
                distribution=dist_config,
                seed=seed,
            )
        )

        # Slack regime: budget large enough that the constraint never binds.
        scenarios.append(
            Scenario(
                name=f"{name}_slack",
                description=f"{desc} Slack-budget regime (rho = max bid, constraint inactive).",
                valuation=valuation,
                bids=BIDS,
                horizon=HORIZON,
                budget=round(max_bid * HORIZON, 2),
                distribution=dist_config,
                seed=seed + 100,
            )
        )
    return scenarios


# Multi-campaign (Requirement 2) configurations.
# (name, description, valuations, distribution-configs, conflict-edges, correlation)
MULTI_CONFIGS = [
    (
        "multi_independent",
        "Three independent campaigns (no conflicts): weak, neutral and strong competition.",
        [0.8, 0.8, 0.9],
        [
            {"type": "truncnorm", "mean": 0.2, "std": 0.1, "low": 0.0, "high": 1.0},
            {"type": "uniform", "low": 0.0, "high": 1.0},
            {"type": "truncnorm", "mean": 0.7, "std": 0.1, "low": 0.0, "high": 1.0},
        ],
        [],
        0.0,
    ),
    (
        "multi_conflict",
        "Four campaigns with a conflict graph: 0-1 (e.g. Coca-Cola/Pepsi) and 2-3 are mutually exclusive.",
        [0.8, 0.75, 0.9, 0.6],
        [
            {"type": "uniform", "low": 0.0, "high": 1.0},
            {"type": "truncnorm", "mean": 0.4, "std": 0.12, "low": 0.0, "high": 1.0},
            {"type": "beta", "a": 2.0, "b": 5.0},
            {"type": "truncnorm", "mean": 0.6, "std": 0.12, "low": 0.0, "high": 1.0},
        ],
        [[0, 1], [2, 3]],
        0.0,
    ),
    (
        "multi_correlated",
        "Three campaigns with positively correlated competing bids (Gaussian copula, rho=0.6).",
        [0.85, 0.8, 0.8],
        [
            {"type": "uniform", "low": 0.0, "high": 1.0},
            {"type": "beta", "a": 2.0, "b": 5.0},
            {
                "type": "bimodal",
                "weight_low": 0.6,
                "mean_low": 0.2,
                "std_low": 0.07,
                "mean_high": 0.75,
                "std_high": 0.07,
            },
        ],
        [],
        0.6,
    ),
]


def build_multi_campaign_scenarios() -> list[MultiCampaignScenario]:
    """Multi-campaign scenarios, each in a budget-binding and a slack regime."""
    scenarios: list[MultiCampaignScenario] = []
    max_bid = max(BIDS)
    for seed, (name, desc, valuations, dists, edges, corr) in enumerate(MULTI_CONFIGS):
        cg = ConflictGraph(len(valuations), [(i, j) for i, j in edges])
        max_active = cg.max_independent_set_size()

        # Unconstrained per-round spend = best feasible bids ignoring the budget.
        probe = MultiCampaignScenario(
            name=name, description=desc, valuations=valuations, bids=BIDS,
            horizon=HORIZON, budget=max_active * max_bid * HORIZON,
            distributions=dists, conflict_edges=edges, correlation=corr, seed=seed,
        )
        unconstrained_cost = multi_campaign_opt(probe).expected_cost

        rho_binding = max(0.5 * unconstrained_cost, BID_STEP)
        scenarios.append(
            MultiCampaignScenario(
                name=f"{name}_binding",
                description=f"{desc} Budget-binding regime (rho approx {rho_binding:.3f}).",
                valuations=valuations, bids=BIDS, horizon=HORIZON,
                budget=round(rho_binding * HORIZON, 2),
                distributions=dists, conflict_edges=edges, correlation=corr, seed=seed,
            )
        )
        scenarios.append(
            MultiCampaignScenario(
                name=f"{name}_slack",
                description=f"{desc} Slack-budget regime (constraint inactive).",
                valuations=valuations, bids=BIDS, horizon=HORIZON,
                budget=round(max_active * max_bid * HORIZON, 2),
                distributions=dists, conflict_edges=edges, correlation=corr, seed=seed + 200,
            )
        )
    return scenarios


# --------------------------------------------------------------------------- #
# Non-stationary multi-campaign (Requirement 3) configurations.
# --------------------------------------------------------------------------- #
def _tn(mean: float, std: float = 0.08) -> dict:
    return {"type": "truncnorm", "mean": mean, "std": std, "low": 0.0, "high": 1.0}


# (name, description, valuations, schedule, conflict-edges)
NONSTATIONARY_CONFIGS = [
    (
        "ns_abrupt",
        "Three campaigns, abruptly changing competition (5 phases): a new "
        "competitor enters/leaves so the best bid jumps between phases.",
        [0.8, 0.8, 0.9],
        {
            "kind": "abrupt",
            "phase_length": HORIZON // 5,
            "campaign_phase_dists": [
                [_tn(0.20), _tn(0.60), _tn(0.30), _tn(0.70), _tn(0.25)],
                [_tn(0.50), _tn(0.25), _tn(0.65), _tn(0.30), _tn(0.55)],
                [_tn(0.70), _tn(0.40), _tn(0.50), _tn(0.20), _tn(0.60)],
            ],
        },
        [],
    ),
    (
        "ns_smooth",
        "Three campaigns whose competition drifts smoothly (sinusoidal mean): "
        "competitors fade in and out continuously over the horizon.",
        [0.85, 0.8, 0.8],
        {
            "kind": "smooth",
            "smooth": [
                {"center": 0.45, "amp": 0.28, "freq": 1.0, "phase": 0.0, "std": 0.07},
                {"center": 0.50, "amp": 0.30, "freq": 2.0, "phase": 1.5, "std": 0.07},
                {"center": 0.40, "amp": 0.25, "freq": 1.5, "phase": 3.0, "std": 0.07},
            ],
        },
        [],
    ),
    (
        "ns_fast",
        "Three campaigns with a distribution that changes quickly over time "
        "(fresh mean every 50 rounds): the highly non-stationary regime.",
        [0.8, 0.85, 0.9],
        {
            "kind": "fast",
            "block_length": 50,
            "low": 0.15,
            "high": 0.75,
            "std": 0.05,
        },
        [],
    ),
    (
        "ns_conflict",
        "Four campaigns with a conflict graph (0-1, 2-3) and abrupt phases that "
        "flip which campaign of each pair is the better buy, forcing switches.",
        [0.8, 0.8, 0.9, 0.85],
        {
            "kind": "abrupt",
            "phase_length": HORIZON // 4,
            "campaign_phase_dists": [
                [_tn(0.25), _tn(0.65), _tn(0.30), _tn(0.60)],
                [_tn(0.60), _tn(0.25), _tn(0.55), _tn(0.30)],
                [_tn(0.30), _tn(0.55), _tn(0.20), _tn(0.65)],
                [_tn(0.55), _tn(0.30), _tn(0.60), _tn(0.25)],
            ],
        },
        [[0, 1], [2, 3]],
    ),
    # Dedicated piecewise-stationary instances for Requirement 4: a "slightly"
    # non-stationary one (few intervals) and a more frequently changing one.
    (
        "pw_slight",
        "Three campaigns, only 3 long intervals (slightly non-stationary): the "
        "ideal regime for change detection / sliding window.",
        [0.8, 0.85, 0.9],
        {
            "kind": "abrupt",
            "phase_length": HORIZON // 3,
            "campaign_phase_dists": [
                [_tn(0.20), _tn(0.60), _tn(0.35)],
                [_tn(0.55), _tn(0.25), _tn(0.60)],
                [_tn(0.65), _tn(0.40), _tn(0.20)],
            ],
        },
        [],
    ),
    (
        "pw_frequent",
        "Three campaigns with 8 shorter intervals (frequently changing): stresses "
        "the window size and the change-detection sensitivity.",
        [0.8, 0.8, 0.9],
        {
            "kind": "abrupt",
            "phase_length": HORIZON // 8,
            "campaign_phase_dists": [
                [_tn(0.20), _tn(0.60), _tn(0.25), _tn(0.65), _tn(0.30), _tn(0.55), _tn(0.20), _tn(0.60)],
                [_tn(0.60), _tn(0.25), _tn(0.60), _tn(0.30), _tn(0.55), _tn(0.25), _tn(0.60), _tn(0.30)],
                [_tn(0.40), _tn(0.55), _tn(0.35), _tn(0.50), _tn(0.60), _tn(0.30), _tn(0.45), _tn(0.55)],
            ],
        },
        [],
    ),
]


def build_nonstationary_scenarios() -> list[NonStationaryMultiCampaignScenario]:
    """Non-stationary multi-campaign scenarios (binding + slack budget regimes).

    The budget regime is derived from the *realized* sequence: we compute the
    best fixed feasible spend in hindsight without a budget and set the binding
    per-round budget to half of it, so the constraint genuinely activates.
    """
    scenarios: list[NonStationaryMultiCampaignScenario] = []
    bids = np.asarray(BIDS, dtype=float)
    max_bid = max(BIDS)
    for seed, (name, desc, valuations, schedule, edges) in enumerate(NONSTATIONARY_CONFIGS):
        cg = ConflictGraph(len(valuations), [(i, j) for i, j in edges])
        max_active = cg.max_independent_set_size()

        probe = NonStationaryMultiCampaignScenario(
            name=name, description=desc, valuations=valuations, bids=BIDS,
            horizon=HORIZON, budget=0.0, schedule=schedule,
            conflict_edges=edges, seed=300 + seed,
        )
        sequence = probe.make_sequence()
        # Unconstrained hindsight spend (rho effectively infinite).
        unconstrained = hindsight_multi_campaign_opt(
            np.asarray(valuations, dtype=float), bids, sequence,
            rho=max_active * max_bid, conflict_edges=[(i, j) for i, j in edges],
        )
        rho_binding = max(0.5 * unconstrained.expected_cost, BID_STEP)

        scenarios.append(
            NonStationaryMultiCampaignScenario(
                name=f"{name}_binding",
                description=f"{desc} Budget-binding regime (rho approx {rho_binding:.3f}).",
                valuations=valuations, bids=BIDS, horizon=HORIZON,
                budget=round(rho_binding * HORIZON, 2), schedule=schedule,
                conflict_edges=edges, seed=300 + seed,
            )
        )
        scenarios.append(
            NonStationaryMultiCampaignScenario(
                name=f"{name}_slack",
                description=f"{desc} Slack-budget regime (constraint inactive).",
                valuations=valuations, bids=BIDS, horizon=HORIZON,
                budget=round(max_active * max_bid * HORIZON, 2), schedule=schedule,
                conflict_edges=edges, seed=300 + seed,
            )
        )
    return scenarios


def main() -> None:
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    single = build_single_scenarios()
    for scenario in single:
        scenario.save(SCENARIO_DIR / f"{scenario.name}.json")
        # Materialize one reproducible m_t sequence for inspection.
        rng = np.random.default_rng(scenario.seed)
        m = scenario.make_distribution().sample(scenario.horizon, rng)
        np.savetxt(
            SAMPLE_DIR / f"{scenario.name}.csv",
            m,
            fmt="%.6f",
            header="competing_bid_m",
            comments="",
        )
        print(
            f"  wrote {scenario.name}: v={scenario.valuation}, "
            f"budget={scenario.budget}, rho={scenario.rho:.4f}"
        )

    multi = build_multi_campaign_scenarios()
    for scenario in multi:
        scenario.save(SCENARIO_DIR / f"{scenario.name}.json")
        # Materialize one reproducible joint m sequence (T, N) for inspection.
        rng = np.random.default_rng(scenario.seed)
        m = scenario.make_joint_distribution().sample(scenario.horizon, rng)
        header = ",".join(f"m_campaign_{i}" for i in range(scenario.n_campaigns))
        np.savetxt(
            SAMPLE_DIR / f"{scenario.name}.csv", m, fmt="%.6f",
            header=header, comments="", delimiter=",",
        )
        print(
            f"  wrote {scenario.name}: N={scenario.n_campaigns}, "
            f"budget={scenario.budget}, rho={scenario.rho:.4f}, corr={scenario.correlation}"
        )

    nonstationary = build_nonstationary_scenarios()
    for scenario in nonstationary:
        scenario.save(SCENARIO_DIR / f"{scenario.name}.json")
        # Materialize the fixed (non-stochastic) joint m sequence (T, N).
        m = scenario.make_sequence()
        header = ",".join(f"m_campaign_{i}" for i in range(scenario.n_campaigns))
        np.savetxt(
            SAMPLE_DIR / f"{scenario.name}.csv", m, fmt="%.6f",
            header=header, comments="", delimiter=",",
        )
        print(
            f"  wrote {scenario.name}: N={scenario.n_campaigns}, kind={scenario.schedule['kind']}, "
            f"budget={scenario.budget}, rho={scenario.rho:.4f}"
        )

    print(
        f"\nGenerated {len(single)} single-campaign, {len(multi)} multi-campaign "
        f"and {len(nonstationary)} non-stationary scenarios in {SCENARIO_DIR}"
    )


if __name__ == "__main__":
    main()
