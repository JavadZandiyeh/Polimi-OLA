"""Requirement 2 experiments: Combinatorial-UCB with budget over many campaigns.

For every multi-campaign scenario this runs Combinatorial-UCB over several seeds
and produces, in ``experiments/plots/``:

* ``regret_<scenario>.png`` -- cumulative pseudo-regret against the combinatorial
  clairvoyant ``T * OPT`` (mean +/- standard error), evidence of sublinear regret.
* ``budget_<scenario>.png`` (budget-binding scenarios) -- remaining budget vs the
  ideal pacing line ``B - rho t``.
* ``diagnostics_<scenario>.png`` (budget-binding scenarios) -- per-campaign
  clairvoyant ``OPT`` bid allocation vs the empirical bid frequency learned by
  Combinatorial-UCB.
* ``scaling_regret_multi.png`` -- final regret vs horizon ``T`` on a
  representative scenario.

It also writes ``experiments/summary_req2.csv``.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CACHE = ROOT / ".mplcache"
_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))

from ola.algorithms import CombinatorialUCBBidder  # noqa: E402
from ola.baseline import multi_campaign_opt  # noqa: E402
from ola.environment import MultiCampaignScenario, MultiCampaignStochasticEnv  # noqa: E402
from ola.metrics import average_curves, regret_curve  # noqa: E402
from ola.simulation import run_multi_episode  # noqa: E402

SCENARIO_DIR = ROOT / "data" / "scenarios"
PLOT_DIR = ROOT / "experiments" / "plots"
SUMMARY_PATH = ROOT / "experiments" / "summary_req2.csv"
N_SEEDS = 5


def load_multi_scenarios() -> list[MultiCampaignScenario]:
    return [
        MultiCampaignScenario.load(p)
        for p in sorted(SCENARIO_DIR.glob("multi_*.json"))
    ]


def run_seeds(scenario: MultiCampaignScenario):
    utilities, budgets = [], []
    bid_counts = np.zeros((scenario.n_campaigns, scenario.bid_array().size))
    stopped_rounds, total_costs = [], []
    for seed in range(N_SEEDS):
        env = MultiCampaignStochasticEnv(scenario, np.random.default_rng(2000 + seed))
        bidder = CombinatorialUCBBidder(
            scenario.valuation_array(), scenario.bid_array(), scenario.horizon,
            scenario.rho, scenario.edge_list(), np.random.default_rng(6000 + seed),
        )
        result = run_multi_episode(env, bidder, scenario.budget)
        utilities.append(result.utilities)
        budgets.append(result.remaining_budget)
        bid_counts += result.bid_counts
        stopped_rounds.append(result.stopped_round)
        total_costs.append(result.total_cost)
    return {
        "utilities": utilities,
        "budgets": budgets,
        "bid_freq": bid_counts / max(bid_counts.sum(), 1.0),
        "mean_stop": float(np.mean(stopped_rounds)),
        "mean_total_util": float(np.mean([u.sum() for u in utilities])),
        "mean_total_cost": float(np.mean(total_costs)),
    }


def plot_regret(scenario: MultiCampaignScenario, opt_value: float, runs) -> None:
    rounds = np.arange(1, scenario.horizon + 1)
    curves = [regret_curve(u, opt_value) for u in runs["utilities"]]
    mean, err = average_curves(curves)
    plt.figure(figsize=(7, 4.5))
    plt.plot(rounds, mean, color="tab:blue", label="Combinatorial-UCB")
    plt.fill_between(rounds, mean - err, mean + err, color="tab:blue", alpha=0.2)
    plt.xlabel("round t")
    plt.ylabel(r"cumulative regret  $t\cdot OPT - \sum f$")
    plt.title(f"Regret vs combinatorial OPT - {scenario.name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"regret_{scenario.name}.png", dpi=120)
    plt.close()


def plot_budget(scenario: MultiCampaignScenario, runs) -> None:
    rounds = np.arange(1, scenario.horizon + 1)
    mean_budget, _ = average_curves(runs["budgets"])
    pace = scenario.budget - scenario.rho * rounds
    plt.figure(figsize=(7, 4.5))
    plt.plot(rounds, mean_budget, color="tab:blue", label="Combinatorial-UCB")
    plt.plot(rounds, pace, color="gray", ls="--", label=r"ideal pacing ($B-\rho t$)")
    plt.xlabel("round t")
    plt.ylabel("remaining budget")
    plt.title(f"Budget depletion - {scenario.name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"budget_{scenario.name}.png", dpi=120)
    plt.close()


def plot_diagnostics(scenario: MultiCampaignScenario, runs) -> None:
    bids = scenario.bid_array()
    opt = multi_campaign_opt(scenario)
    n = scenario.n_campaigns
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    width = 0.4 * (bids[1] - bids[0])
    for i in range(n):
        ax = axes[0][i]
        ax.bar(bids - width / 2, opt.x[i], width=width, label="OPT", color="tab:blue")
        ax.bar(bids + width / 2, runs["bid_freq"][i], width=width, label="learned", color="tab:cyan")
        ax.set_title(f"campaign {i} (v={scenario.valuations[i]})")
        ax.set_xlabel("bid b")
        if i == 0:
            ax.set_ylabel("probability")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Per-campaign bid allocation - {scenario.name}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"diagnostics_{scenario.name}.png", dpi=120)
    plt.close(fig)


def plot_scaling(base: MultiCampaignScenario) -> None:
    horizons = [2000, 5000, 10000, 20000, 40000]
    means, errs = [], []
    for T in horizons:
        scenario = replace(base, horizon=T, budget=round(base.rho * T, 2))
        opt = multi_campaign_opt(scenario)
        finals = []
        for seed in range(N_SEEDS):
            env = MultiCampaignStochasticEnv(scenario, np.random.default_rng(8000 + seed))
            bidder = CombinatorialUCBBidder(
                scenario.valuation_array(), scenario.bid_array(), T, scenario.rho,
                scenario.edge_list(), np.random.default_rng(9500 + seed),
            )
            result = run_multi_episode(env, bidder, scenario.budget)
            finals.append(T * opt.value - result.total_utility)
        means.append(np.mean(finals))
        errs.append(np.std(finals, ddof=1) / np.sqrt(N_SEEDS))

    h = np.array(horizons, dtype=float)
    means_arr = np.array(means)
    ref = means_arr[-1] * np.sqrt(h / h[-1])
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(h, means_arr, yerr=errs, marker="o", label="Combinatorial-UCB final regret")
    plt.plot(h, ref, ls="--", color="gray", label=r"$\propto\sqrt{T}$ reference")
    plt.xlabel("horizon T")
    plt.ylabel(r"final regret  $T\cdot OPT - \sum f$")
    plt.title(f"Regret scaling with T - {base.name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "scaling_regret_multi.png", dpi=120)
    plt.close()


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = load_multi_scenarios()
    if not scenarios:
        raise SystemExit("No multi-campaign scenarios found. Run data/generate_datasets.py first.")

    rows = []
    for scenario in scenarios:
        opt = multi_campaign_opt(scenario)
        is_binding = scenario.name.endswith("_binding")
        print(
            f"[{scenario.name}] N={scenario.n_campaigns} rho={scenario.rho:.3f} "
            f"OPT/round={opt.value:.4f} corr={scenario.correlation}"
        )
        runs = run_seeds(scenario)
        plot_regret(scenario, opt.value, runs)
        if is_binding:
            plot_budget(scenario, runs)
            plot_diagnostics(scenario, runs)

        rows.append(
            {
                "scenario": scenario.name,
                "n_campaigns": scenario.n_campaigns,
                "correlation": scenario.correlation,
                "rho": round(scenario.rho, 4),
                "opt_per_round": round(opt.value, 4),
                "T_OPT": round(scenario.horizon * opt.value, 1),
                "cucb_total_util": round(runs["mean_total_util"], 1),
                "cucb_total_cost": round(runs["mean_total_cost"], 1),
                "cucb_stop_round": round(runs["mean_stop"], 0),
            }
        )
        print(
            f"    Comb-UCB util={runs['mean_total_util']:.0f} "
            f"(stop {runs['mean_stop']:.0f}) cost={runs['mean_total_cost']:.0f} "
            f"budget={scenario.budget:.0f} | T*OPT={scenario.horizon * opt.value:.0f}"
        )

    print("\nScaling experiment (regret vs T) on multi_conflict_binding ...")
    base = next(s for s in scenarios if s.name == "multi_conflict_binding")
    plot_scaling(base)

    with SUMMARY_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved plots to {PLOT_DIR} and summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
