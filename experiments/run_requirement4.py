"""Requirement 4 experiments: slightly non-stationary multiple campaigns.

On *piecewise-stationary* multi-campaign environments (the horizon is split into
intervals, each with its own fixed distribution of highest competing bids) we
compare four bidders:

* **Comb-UCB** (req2) -- the stationary baseline, which adapts slowly and tends
  to deplete its budget early once a change makes its estimates stale;
* **SW-CombUCB** (req4) -- Combinatorial-UCB with a sliding window (passive
  forgetting);
* **CD-CombUCB** (req4) -- Combinatorial-UCB with a CUSUM change detector
  (active reset);
* **Primal-Dual** (req3) -- the best-of-both-worlds primal-dual bidder.

Regret is measured against the **dynamic per-interval optimum**
(:func:`ola.baseline.piecewise_dynamic_opt`): the clairvoyant policy that knows
the intervals and plays each interval's own constrained ``OPT``.  This is the
yardstick that sliding-window / change-detection methods aim to track.

Outputs in ``experiments/plots/``:

* ``regret_req4_<scenario>.png`` -- cumulative regret of all four bidders vs the
  dynamic optimum, with interval boundaries marked.
* ``budget_req4_<scenario>.png`` -- remaining budget of all four bidders.
* ``window_sensitivity.png`` -- SW-CombUCB final regret vs window size ``W``
  (illustrating that the best ``W`` depends on the change frequency).
* ``req4_summary.png`` -- final regret of every bidder across all scenarios.

It also writes ``experiments/summary_req4.csv``.
"""

from __future__ import annotations

import csv
import os
import sys
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

from ola.algorithms import (  # noqa: E402
    ChangeDetectionCombinatorialUCBBidder,
    CombinatorialUCBBidder,
    PrimalDualBidder,
    SlidingWindowCombinatorialUCBBidder,
)
from ola.baseline import hindsight_multi_campaign_opt, piecewise_dynamic_opt  # noqa: E402
from ola.environment import (  # noqa: E402
    NonStationaryMultiCampaignEnv,
    NonStationaryMultiCampaignScenario,
)
from ola.metrics import average_curves  # noqa: E402
from ola.simulation import run_multi_episode, run_primal_dual_episode  # noqa: E402

SCENARIO_DIR = ROOT / "data" / "scenarios"
PLOT_DIR = ROOT / "experiments" / "plots"
SUMMARY_PATH = ROOT / "experiments" / "summary_req4.csv"

# Piecewise-stationary scenarios (req4), ordered by change frequency.
SCENARIOS = [
    "pw_slight_binding",      # 3 intervals  (slightly non-stationary)
    "ns_conflict_binding",    # 4 intervals  (+ conflict graph)
    "ns_abrupt_binding",      # 5 intervals
    "pw_frequent_binding",    # 8 intervals  (frequently changing)
]
N_SEEDS = 4
SENS_SEEDS = 2

COLORS = {
    "Comb-UCB": "tab:gray",
    "SW-CombUCB": "tab:green",
    "CD-CombUCB": "tab:orange",
    "Primal-Dual": "tab:blue",
}


# --------------------------------------------------------------------------- #
def dynamic_baseline_curve(scenario, dyn) -> np.ndarray:
    """Per-round cumulative benchmark from the per-interval OPT values."""
    phase_len = int(scenario.schedule["phase_length"])
    horizon = scenario.horizon
    rate = np.empty(horizon)
    for p, val in enumerate(dyn.interval_values):
        t0, t1 = p * phase_len, min((p + 1) * phase_len, horizon)
        rate[t0:t1] = val
    return np.cumsum(rate)


def regret_curves_vs_dynamic(utilities_runs, baseline_cum) -> list[np.ndarray]:
    return [baseline_cum - np.cumsum(u) for u in utilities_runs]


def make_bidders(scenario):
    """Factories for the four bidders (fresh RNG per seed)."""
    va, bi = scenario.valuation_array(), scenario.bid_array()
    ho, rho, ed = scenario.horizon, scenario.rho, scenario.edge_list()

    def comb(seed):
        return CombinatorialUCBBidder(va, bi, ho, rho, ed, np.random.default_rng(6000 + seed))

    def sw(seed):
        return SlidingWindowCombinatorialUCBBidder(va, bi, ho, rho, ed, np.random.default_rng(6000 + seed))

    def cd(seed):
        return ChangeDetectionCombinatorialUCBBidder(va, bi, ho, rho, ed, np.random.default_rng(6000 + seed))

    def pd(seed):
        return PrimalDualBidder(va, bi, ho, rho, ed, np.random.default_rng(7000 + seed))

    return {"Comb-UCB": comb, "SW-CombUCB": sw, "CD-CombUCB": cd, "Primal-Dual": pd}


def run_bidder(scenario, factory, is_pd, n_seeds=N_SEEDS):
    utilities, budgets, stops, costs = [], [], [], []
    for seed in range(n_seeds):
        env = NonStationaryMultiCampaignEnv(scenario)
        bidder = factory(seed)
        runner = run_primal_dual_episode if is_pd else run_multi_episode
        result = runner(env, bidder, scenario.budget)
        utilities.append(result.utilities)
        budgets.append(result.remaining_budget)
        stops.append(result.stopped_round)
        costs.append(result.total_cost)
    return {
        "utilities": utilities,
        "budgets": budgets,
        "mean_total_util": float(np.mean([u.sum() for u in utilities])),
        "mean_stop": float(np.mean(stops)),
        "mean_total_cost": float(np.mean(costs)),
    }


# --------------------------------------------------------------------------- #
def plot_regret(scenario, baseline_cum, runs_by_algo) -> None:
    rounds = np.arange(1, scenario.horizon + 1)
    phase_len = int(scenario.schedule["phase_length"])
    plt.figure(figsize=(7.5, 4.8))
    for algo, runs in runs_by_algo.items():
        curves = regret_curves_vs_dynamic(runs["utilities"], baseline_cum)
        mean, err = average_curves(curves)
        plt.plot(rounds, mean, color=COLORS[algo], label=algo)
        plt.fill_between(rounds, mean - err, mean + err, color=COLORS[algo], alpha=0.15)
    for b in range(phase_len, scenario.horizon, phase_len):
        plt.axvline(b, color="black", lw=0.6, ls=":", alpha=0.5)
    plt.xlabel("round t")
    plt.ylabel(r"cumulative regret  $\sum_t OPT^{dyn}_t - \sum f$")
    plt.title(f"Regret vs dynamic per-interval optimum - {scenario.name}")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"regret_req4_{scenario.name}.png", dpi=120)
    plt.close()


def plot_budget(scenario, runs_by_algo) -> None:
    rounds = np.arange(1, scenario.horizon + 1)
    phase_len = int(scenario.schedule["phase_length"])
    plt.figure(figsize=(7.5, 4.8))
    for algo, runs in runs_by_algo.items():
        mean_budget, _ = average_curves(runs["budgets"])
        plt.plot(rounds, mean_budget, color=COLORS[algo], label=algo)
    pace = scenario.budget - scenario.rho * rounds
    plt.plot(rounds, pace, color="black", ls="--", lw=0.9, label=r"ideal pacing ($B-\rho t$)")
    for b in range(phase_len, scenario.horizon, phase_len):
        plt.axvline(b, color="black", lw=0.6, ls=":", alpha=0.4)
    plt.xlabel("round t")
    plt.ylabel("remaining budget")
    plt.title(f"Budget depletion - {scenario.name}")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"budget_req4_{scenario.name}.png", dpi=120)
    plt.close()


def plot_window_sensitivity(scenario) -> None:
    va, bi = scenario.valuation_array(), scenario.bid_array()
    ho, rho, ed = scenario.horizon, scenario.rho, scenario.edge_list()
    dyn = piecewise_dynamic_opt(scenario)
    D = dyn.total_value
    windows = [250, 500, 1000, 2000, 3000, 4000]
    means, errs = [], []
    for W in windows:
        finals = []
        for seed in range(SENS_SEEDS):
            env = NonStationaryMultiCampaignEnv(scenario)
            bidder = SlidingWindowCombinatorialUCBBidder(
                va, bi, ho, rho, ed, np.random.default_rng(6000 + seed), window=W
            )
            result = run_multi_episode(env, bidder, scenario.budget)
            finals.append(D - result.total_utility)
        means.append(np.mean(finals))
        errs.append(np.std(finals, ddof=1) / np.sqrt(SENS_SEEDS) if SENS_SEEDS > 1 else 0.0)
    interval_len = int(scenario.schedule["phase_length"])
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(windows, means, yerr=errs, marker="o", color="tab:green", label="SW-CombUCB final regret")
    plt.axvline(interval_len, color="gray", ls="--", label=f"interval length = {interval_len}")
    plt.xlabel("sliding window size W")
    plt.ylabel("final regret vs dynamic optimum")
    plt.title(f"Sliding-window sensitivity - {scenario.name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "window_sensitivity.png", dpi=120)
    plt.close()


def plot_summary(rows) -> None:
    scenarios = [r["scenario"] for r in rows]
    algos = ["Comb-UCB", "SW-CombUCB", "CD-CombUCB", "Primal-Dual"]
    x = np.arange(len(scenarios))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(scenarios)), 5))
    for k, algo in enumerate(algos):
        vals = [r[f"regret_{algo}"] for r in rows]
        ax.bar(x + (k - 1.5) * width, vals, width, label=algo, color=COLORS[algo])
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_binding", "") for s in scenarios], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("final regret vs dynamic optimum")
    ax.set_title("Requirement 4: tracking a piecewise-stationary environment")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "req4_summary.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in SCENARIOS:
        scenario = NonStationaryMultiCampaignScenario.load(SCENARIO_DIR / f"{name}.json")
        dyn = piecewise_dynamic_opt(scenario)
        baseline_cum = dynamic_baseline_curve(scenario, dyn)
        fixed = hindsight_multi_campaign_opt(
            scenario.valuation_array(), scenario.bid_array(), scenario.make_sequence(),
            scenario.rho, scenario.edge_list(),
        )
        D = dyn.total_value

        factories = make_bidders(scenario)
        runs_by_algo = {}
        for algo, factory in factories.items():
            runs_by_algo[algo] = run_bidder(scenario, factory, is_pd=(algo == "Primal-Dual"))

        plot_regret(scenario, baseline_cum, runs_by_algo)
        plot_budget(scenario, runs_by_algo)

        row = {
            "scenario": scenario.name,
            "n_intervals": dyn.n_intervals,
            "n_campaigns": scenario.n_campaigns,
            "rho": round(scenario.rho, 4),
            "T_OPT_dynamic": round(D, 1),
            "T_OPT_fixed": round(scenario.horizon * fixed.value, 1),
        }
        print(f"[{scenario.name}] intervals={dyn.n_intervals} dyn-OPT={D:.0f} "
              f"fixed-OPT={scenario.horizon * fixed.value:.0f}")
        for algo, runs in runs_by_algo.items():
            reg = D - runs["mean_total_util"]
            row[f"regret_{algo}"] = round(reg, 1)
            row[f"util_{algo}"] = round(runs["mean_total_util"], 1)
            row[f"stop_{algo}"] = round(runs["mean_stop"], 0)
            print(f"    {algo:12s} util={runs['mean_total_util']:6.0f} "
                  f"regret={reg:6.0f} stop={runs['mean_stop']:5.0f} "
                  f"cost={runs['mean_total_cost']:.0f}/{scenario.budget:.0f}")
        rows.append(row)

    plot_summary(rows)

    print("\nWindow-sensitivity sweep on pw_slight_binding ...")
    sens = NonStationaryMultiCampaignScenario.load(SCENARIO_DIR / "pw_slight_binding.json")
    plot_window_sensitivity(sens)

    with SUMMARY_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved plots to {PLOT_DIR} and summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
