"""Requirement 1 experiments: UCB1 vs budget-aware UCB on a single campaign.

For every synthesized scenario this script runs both algorithms over several
seeds and produces, in ``experiments/plots/``:

* ``regret_<scenario>.png`` -- cumulative pseudo-regret of both algorithms
  against the constrained clairvoyant ``T * OPT`` (mean +/- standard error).
* ``budget_<scenario>.png`` (budget-binding scenarios) -- remaining budget over
  time, showing Algorithm A draining its budget early while Algorithm B paces.
* ``diagnostics_<scenario>.png`` (budget-binding scenarios) -- per-bid expected
  utility/cost, the clairvoyant ``OPT`` mixture and the empirical bid frequency
  of Algorithm B.
* ``scaling_regret.png`` -- final regret of Algorithm B vs horizon ``T``,
  evidence of the ``O(sqrt(T))`` (sublinear) regret.

It also writes ``experiments/summary.csv`` with the headline numbers.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import replace
from pathlib import Path

# Use a local, writable matplotlib cache (avoids a noisy warning when the
# default ~/.matplotlib is not writable) and a non-interactive backend.
_CACHE = Path(__file__).resolve().parents[1] / ".mplcache"
_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ola.algorithms.ucb1 import UCB1Bidder  # noqa: E402
from ola.algorithms.ucb_budget import BudgetedUCBBidder  # noqa: E402
from ola.baseline import (  # noqa: E402
    best_fixed_bid,
    constrained_opt,
    expected_cost_per_bid,
    expected_utility_per_bid,
)
from ola.environment import Scenario, StochasticSingleCampaignEnv  # noqa: E402
from ola.metrics import average_curves, regret_curve  # noqa: E402
from ola.simulation import run_episode  # noqa: E402

SCENARIO_DIR = ROOT / "data" / "scenarios"
PLOT_DIR = ROOT / "experiments" / "plots"
SUMMARY_PATH = ROOT / "experiments" / "summary.csv"
N_SEEDS = 5


def load_single_scenarios() -> list[Scenario]:
    scenarios = []
    for path in sorted(SCENARIO_DIR.glob("*.json")):
        try:
            scenarios.append(Scenario.load(path))
        except TypeError:
            # Multi-campaign scenarios have a different schema -- skip here.
            continue
    return scenarios


def run_seeds(scenario: Scenario, algo: str):
    """Run ``N_SEEDS`` episodes; return per-round utility and budget matrices."""
    bids = scenario.bid_array()
    utilities, budgets, bid_counts = [], [], np.zeros(bids.size)
    stopped_rounds = []
    for seed in range(N_SEEDS):
        env = StochasticSingleCampaignEnv(scenario, np.random.default_rng(1000 + seed))
        if algo == "ucb1":
            bidder = UCB1Bidder(bids, scenario.valuation, scenario.horizon)
        else:
            bidder = BudgetedUCBBidder(
                bids, scenario.valuation, scenario.horizon, scenario.rho,
                np.random.default_rng(5000 + seed),
            )
        result = run_episode(env, bidder, scenario.budget)
        utilities.append(result.utilities)
        budgets.append(result.remaining_budget)
        stopped_rounds.append(result.stopped_round)
        played = result.bid_indices[result.bid_indices >= 0]
        bid_counts += np.bincount(played, minlength=bids.size)
    return {
        "utilities": utilities,
        "budgets": budgets,
        "bid_freq": bid_counts / bid_counts.sum(),
        "mean_stop": float(np.mean(stopped_rounds)),
        "mean_total_util": float(np.mean([u.sum() for u in utilities])),
        "mean_total_cost": float(np.mean([scenario.budget - b[-1] for b in budgets])),
    }


def plot_regret(scenario: Scenario, opt_value: float, ucb1, budgeted) -> None:
    rounds = np.arange(1, scenario.horizon + 1)

    r_ucb1 = [regret_curve(u, opt_value) for u in ucb1["utilities"]]
    r_bud = [regret_curve(u, opt_value) for u in budgeted["utilities"]]
    m1, e1 = average_curves(r_ucb1)
    m2, e2 = average_curves(r_bud)

    plt.figure(figsize=(7, 4.5))
    plt.plot(rounds, m1, label="UCB1 (ignores budget)", color="tab:red")
    plt.fill_between(rounds, m1 - e1, m1 + e1, color="tab:red", alpha=0.2)
    plt.plot(rounds, m2, label="Budgeted UCB", color="tab:blue")
    plt.fill_between(rounds, m2 - e2, m2 + e2, color="tab:blue", alpha=0.2)
    plt.xlabel("round t")
    plt.ylabel(r"cumulative regret  $t\cdot OPT - \sum f$")
    plt.title(f"Regret vs constrained OPT - {scenario.name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"regret_{scenario.name}.png", dpi=120)
    plt.close()


def plot_budget(scenario: Scenario, ucb1, budgeted) -> None:
    rounds = np.arange(1, scenario.horizon + 1)
    b1, _ = average_curves(ucb1["budgets"])
    b2, _ = average_curves(budgeted["budgets"])
    pace = scenario.budget - scenario.rho * rounds  # ideal constant-rate spend

    plt.figure(figsize=(7, 4.5))
    plt.plot(rounds, b1, label="UCB1 (ignores budget)", color="tab:red")
    plt.plot(rounds, b2, label="Budgeted UCB", color="tab:blue")
    plt.plot(rounds, pace, label=r"ideal pacing ($B-\rho t$)", color="gray", ls="--")
    plt.xlabel("round t")
    plt.ylabel("remaining budget")
    plt.title(f"Budget depletion - {scenario.name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"budget_{scenario.name}.png", dpi=120)
    plt.close()


def plot_diagnostics(scenario: Scenario, budgeted) -> None:
    bids = scenario.bid_array()
    dist = scenario.make_distribution()
    f = expected_utility_per_bid(scenario.valuation, bids, dist)
    c = expected_cost_per_bid(bids, dist)
    opt = constrained_opt(scenario.valuation, bids, dist, scenario.rho)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(bids, f, "o-", label=r"$E[f(b)]$ utility", color="tab:green")
    ax[0].plot(bids, c, "s-", label=r"$E[c(b)]$ cost", color="tab:orange")
    ax[0].axhline(scenario.rho, color="gray", ls="--", label=r"$\rho=B/T$")
    ax[0].set_xlabel("bid b")
    ax[0].set_ylabel("expected value per round")
    ax[0].set_title("Per-bid expected utility / cost")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    width = 0.4 * (bids[1] - bids[0])
    ax[1].bar(bids - width / 2, opt.gamma, width=width, label="OPT mixture", color="tab:blue")
    ax[1].bar(bids + width / 2, budgeted["bid_freq"], width=width, label="Budgeted UCB freq", color="tab:cyan")
    ax[1].set_xlabel("bid b")
    ax[1].set_ylabel("probability")
    ax[1].set_title("Clairvoyant vs learned bid distribution")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    fig.suptitle(f"Diagnostics - {scenario.name}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"diagnostics_{scenario.name}.png", dpi=120)
    plt.close(fig)


def plot_scaling(base: Scenario) -> None:
    horizons = [2000, 5000, 10000, 20000, 40000]
    bids = base.bid_array()
    dist = base.make_distribution()
    means, errs = [], []
    for T in horizons:
        scenario = replace(base, horizon=T, budget=round(base.rho * T, 2))
        opt = constrained_opt(scenario.valuation, bids, dist, scenario.rho)
        finals = []
        for seed in range(N_SEEDS):
            env = StochasticSingleCampaignEnv(scenario, np.random.default_rng(7000 + seed))
            bidder = BudgetedUCBBidder(
                bids, scenario.valuation, T, scenario.rho, np.random.default_rng(9000 + seed)
            )
            result = run_episode(env, bidder, scenario.budget)
            finals.append(T * opt.value - result.total_utility)
        means.append(np.mean(finals))
        errs.append(np.std(finals, ddof=1) / np.sqrt(N_SEEDS))

    horizons_arr = np.array(horizons, dtype=float)
    means_arr = np.array(means)
    # Fit a sqrt(T) reference scaled to the largest horizon.
    ref = means_arr[-1] * np.sqrt(horizons_arr / horizons_arr[-1])

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(horizons_arr, means_arr, yerr=errs, marker="o", label="Budgeted UCB final regret")
    plt.plot(horizons_arr, ref, ls="--", color="gray", label=r"$\propto\sqrt{T}$ reference")
    plt.xlabel("horizon T")
    plt.ylabel("final regret  $T\\cdot OPT - \\sum f$")
    plt.title(f"Regret scaling with T - {base.name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "scaling_regret.png", dpi=120)
    plt.close()


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = load_single_scenarios()
    if not scenarios:
        raise SystemExit("No scenarios found. Run data/generate_datasets.py first.")

    rows = []
    for scenario in scenarios:
        bids = scenario.bid_array()
        dist = scenario.make_distribution()
        opt = constrained_opt(scenario.valuation, bids, dist, scenario.rho)
        bf = best_fixed_bid(scenario.valuation, bids, dist)
        is_binding = scenario.name.endswith("_binding")

        print(f"[{scenario.name}] rho={scenario.rho:.4f}  OPT/round={opt.value:.4f}")
        ucb1 = run_seeds(scenario, "ucb1")
        budgeted = run_seeds(scenario, "budgeted")

        plot_regret(scenario, opt.value, ucb1, budgeted)
        if is_binding:
            plot_budget(scenario, ucb1, budgeted)
            plot_diagnostics(scenario, budgeted)

        rows.append(
            {
                "scenario": scenario.name,
                "valuation": scenario.valuation,
                "rho": round(scenario.rho, 4),
                "T_opt_per_round": round(opt.value, 4),
                "best_fixed_bid": bf.bid,
                "ucb1_total_util": round(ucb1["mean_total_util"], 1),
                "ucb1_stop_round": round(ucb1["mean_stop"], 0),
                "budgeted_total_util": round(budgeted["mean_total_util"], 1),
                "budgeted_total_cost": round(budgeted["mean_total_cost"], 1),
                "budgeted_stop_round": round(budgeted["mean_stop"], 0),
                "T_OPT": round(scenario.horizon * opt.value, 1),
            }
        )
        print(
            f"    UCB1 util={ucb1['mean_total_util']:.0f} (stop {ucb1['mean_stop']:.0f}) | "
            f"Budgeted util={budgeted['mean_total_util']:.0f} "
            f"(stop {budgeted['mean_stop']:.0f}) | T*OPT={scenario.horizon * opt.value:.0f}"
        )

    print("\nScaling experiment (regret vs T) on bimodal_binding ...")
    base = next(s for s in scenarios if s.name == "bimodal_binding")
    plot_scaling(base)

    with SUMMARY_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved plots to {PLOT_DIR} and summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
