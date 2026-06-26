"""Requirement 3 experiments: best-of-both-worlds primal-dual bidding.

The same primal-dual bidder (:class:`ola.algorithms.PrimalDualBidder`) is run on
**both** worlds and compared to the stochastic-only Combinatorial-UCB (req2):

* the **stochastic** multi-campaign scenarios of Requirement 2 (data reused), and
* new **highly non-stationary** multi-campaign scenarios (``ns_*``: abrupt,
  smooth, fast and conflict-graph regimes).

Regret is measured against the best *fixed* feasible bidding distribution in
hindsight (:func:`ola.baseline.hindsight_multi_campaign_opt`) -- the benchmark a
best-of-both-worlds guarantee targets, which reduces to the req2 clairvoyant in
the stochastic limit.  We also report a *dynamic* (piecewise) hindsight benchmark
that is allowed to track the changes, to quantify the price of non-stationarity.

Outputs in ``experiments/plots/``:

* ``regret_<scenario>.png`` -- PD vs Comb-UCB cumulative regret (both worlds).
* ``budget_<ns scenario>.png`` -- PD remaining budget vs ideal pacing ``B-rho t``.
* ``dual_<ns scenario>.png`` -- the dual variable ``lambda_t`` trajectory.
* ``bobw_summary.png`` -- PD vs Comb-UCB final regret across every scenario.
* ``scaling_regret_pd.png`` -- PD final regret vs horizon ``T`` (``~sqrt(T)``).

It also writes ``experiments/summary_req3.csv``.
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

from ola.algorithms import CombinatorialUCBBidder, PrimalDualBidder  # noqa: E402
from ola.baseline import (  # noqa: E402
    hindsight_multi_campaign_opt,
    multi_campaign_opt,
)
from ola.environment import (  # noqa: E402
    MultiCampaignScenario,
    MultiCampaignStochasticEnv,
    NonStationaryMultiCampaignEnv,
    NonStationaryMultiCampaignScenario,
)
from ola.metrics import average_curves, regret_curve  # noqa: E402
from ola.simulation import run_multi_episode, run_primal_dual_episode  # noqa: E402

SCENARIO_DIR = ROOT / "data" / "scenarios"
PLOT_DIR = ROOT / "experiments" / "plots"
SUMMARY_PATH = ROOT / "experiments" / "summary_req3.csv"

PD_SEEDS = 5
CUCB_SEEDS = 3
DYNAMIC_WINDOW = 250  # window length for the piecewise (tracking) benchmark


# --------------------------------------------------------------------------- #
# Benchmarks.
# --------------------------------------------------------------------------- #
def dynamic_total_opt(
    valuations: np.ndarray,
    bids: np.ndarray,
    sequence: np.ndarray,
    rho: float,
    edges,
    window: int = DYNAMIC_WINDOW,
) -> float:
    """Piecewise hindsight optimum that may *change* every ``window`` rounds.

    Sum over windows of (window length) x (constrained hindsight OPT of that
    window).  As ``window -> 1`` this approaches the fully dynamic optimum; the
    gap to the fixed-action OPT is the price of non-stationarity.
    """
    T = sequence.shape[0]
    total = 0.0
    for t0 in range(0, T, window):
        t1 = min(t0 + window, T)
        sol = hindsight_multi_campaign_opt(
            valuations, bids, sequence[t0:t1], rho, edges
        )
        total += (t1 - t0) * sol.value
    return total


# --------------------------------------------------------------------------- #
# Runners.
# --------------------------------------------------------------------------- #
def run_pd(scenario, make_env, n_seeds=PD_SEEDS):
    utilities, budgets, lambdas, stops, costs = [], [], [], [], []
    bid_counts = np.zeros((scenario.n_campaigns, scenario.bid_array().size))
    for seed in range(n_seeds):
        env = make_env(seed)
        bidder = PrimalDualBidder(
            scenario.valuation_array(), scenario.bid_array(), scenario.horizon,
            scenario.rho, scenario.edge_list(), np.random.default_rng(7000 + seed),
        )
        result = run_primal_dual_episode(env, bidder, scenario.budget)
        utilities.append(result.utilities)
        budgets.append(result.remaining_budget)
        lambdas.append(bidder.lambda_history.copy())
        stops.append(result.stopped_round)
        costs.append(result.total_cost)
        bid_counts += result.bid_counts
    return {
        "utilities": utilities,
        "budgets": budgets,
        "lambdas": lambdas,
        "mean_stop": float(np.mean(stops)),
        "mean_total_util": float(np.mean([u.sum() for u in utilities])),
        "mean_total_cost": float(np.mean(costs)),
        "bid_freq": bid_counts / max(bid_counts.sum(), 1.0),
    }


def run_cucb(scenario, make_env, n_seeds=CUCB_SEEDS):
    utilities, stops, costs = [], [], []
    for seed in range(n_seeds):
        env = make_env(seed)
        bidder = CombinatorialUCBBidder(
            scenario.valuation_array(), scenario.bid_array(), scenario.horizon,
            scenario.rho, scenario.edge_list(), np.random.default_rng(6000 + seed),
        )
        result = run_multi_episode(env, bidder, scenario.budget)
        utilities.append(result.utilities)
        stops.append(result.stopped_round)
        costs.append(result.total_cost)
    return {
        "utilities": utilities,
        "mean_stop": float(np.mean(stops)),
        "mean_total_util": float(np.mean([u.sum() for u in utilities])),
        "mean_total_cost": float(np.mean(costs)),
    }


# --------------------------------------------------------------------------- #
# Plots.
# --------------------------------------------------------------------------- #
def plot_regret(scenario, opt_per_round, pd_runs, cucb_runs, title_suffix) -> None:
    rounds = np.arange(1, scenario.horizon + 1)
    plt.figure(figsize=(7, 4.5))
    for runs, color, label in [
        (pd_runs, "tab:blue", "Primal-Dual (best-of-both-worlds)"),
        (cucb_runs, "tab:red", "Combinatorial-UCB (stochastic-only)"),
    ]:
        curves = [regret_curve(u, opt_per_round) for u in runs["utilities"]]
        mean, err = average_curves(curves)
        plt.plot(rounds, mean, color=color, label=label)
        plt.fill_between(rounds, mean - err, mean + err, color=color, alpha=0.18)
    plt.axhline(0.0, color="gray", lw=0.8, ls=":")
    plt.xlabel("round t")
    plt.ylabel(r"cumulative regret  $t\cdot OPT_{fixed} - \sum f$")
    plt.title(f"Regret vs best fixed hindsight - {scenario.name} ({title_suffix})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"regret_{scenario.name}.png", dpi=120)
    plt.close()


def plot_budget(scenario, pd_runs) -> None:
    rounds = np.arange(1, scenario.horizon + 1)
    mean_budget, _ = average_curves(pd_runs["budgets"])
    pace = scenario.budget - scenario.rho * rounds
    plt.figure(figsize=(7, 4.5))
    plt.plot(rounds, mean_budget, color="tab:blue", label="Primal-Dual remaining budget")
    plt.plot(rounds, pace, color="gray", ls="--", label=r"ideal pacing ($B-\rho t$)")
    plt.xlabel("round t")
    plt.ylabel("remaining budget")
    plt.title(f"Budget depletion - {scenario.name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"budget_{scenario.name}.png", dpi=120)
    plt.close()


def plot_dual(scenario, pd_runs) -> None:
    rounds = np.arange(1, scenario.horizon + 1)
    mean_lam, err = average_curves(pd_runs["lambdas"])
    plt.figure(figsize=(7, 4.5))
    plt.plot(rounds, mean_lam, color="tab:purple", label=r"dual variable $\lambda_t$")
    plt.fill_between(rounds, mean_lam - err, mean_lam + err, color="tab:purple", alpha=0.18)
    plt.xlabel("round t")
    plt.ylabel(r"$\lambda_t$ (budget price)")
    plt.title(f"Dual variable trajectory - {scenario.name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"dual_{scenario.name}.png", dpi=120)
    plt.close()


def plot_bobw_summary(rows) -> None:
    labels = [r["scenario"] for r in rows]
    pd_reg = [r["pd_regret"] for r in rows]
    cu_reg = [r["cucb_regret"] for r in rows]
    worlds = [r["world"] for r in rows]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(labels)), 5))
    ax.bar(x - width / 2, pd_reg, width, label="Primal-Dual", color="tab:blue")
    ax.bar(x + width / 2, cu_reg, width, label="Combinatorial-UCB", color="tab:red")
    for i, w in enumerate(worlds):
        ax.annotate(
            "stoch." if w == "stochastic" else "non-stat.",
            (x[i], 0), textcoords="offset points", xytext=(0, -28),
            ha="center", fontsize=7, rotation=0, color="gray",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("final regret vs fixed hindsight")
    ax.set_title("Best-of-both-worlds: one algorithm, both regimes")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "bobw_summary.png", dpi=120)
    plt.close(fig)


def plot_scaling(base: NonStationaryMultiCampaignScenario) -> None:
    horizons = [2000, 5000, 10000, 20000, 40000]
    means, errs = [], []
    for T in horizons:
        # Rescale phase/block lengths with the horizon so the *degree* of
        # non-stationarity is comparable across T.
        sched = dict(base.schedule)
        if sched["kind"] == "fast":
            sched["block_length"] = max(10, int(base.schedule["block_length"] * T / base.horizon))
        elif sched["kind"] == "abrupt":
            sched["phase_length"] = max(1, int(base.schedule["phase_length"] * T / base.horizon))
        scenario = replace(base, horizon=T, budget=round(base.rho * T, 2), schedule=sched)
        seq = scenario.make_sequence()
        opt = hindsight_multi_campaign_opt(
            scenario.valuation_array(), scenario.bid_array(), seq, scenario.rho,
            scenario.edge_list(),
        )
        finals = []
        for seed in range(PD_SEEDS):
            env = NonStationaryMultiCampaignEnv(scenario)
            bidder = PrimalDualBidder(
                scenario.valuation_array(), scenario.bid_array(), T, scenario.rho,
                scenario.edge_list(), np.random.default_rng(9500 + seed),
            )
            result = run_primal_dual_episode(env, bidder, scenario.budget)
            finals.append(T * opt.value - result.total_utility)
        means.append(np.mean(finals))
        errs.append(np.std(finals, ddof=1) / np.sqrt(PD_SEEDS))

    h = np.array(horizons, dtype=float)
    means_arr = np.array(means)
    ref = means_arr[-1] * np.sqrt(h / h[-1])
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(h, means_arr, yerr=errs, marker="o", color="tab:blue", label="Primal-Dual final regret")
    plt.plot(h, ref, ls="--", color="gray", label=r"$\propto\sqrt{T}$ reference")
    plt.xlabel("horizon T")
    plt.ylabel(r"final regret  $T\cdot OPT_{fixed} - \sum f$")
    plt.title(f"Regret scaling with T - {base.name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "scaling_regret_pd.png", dpi=120)
    plt.close()


# --------------------------------------------------------------------------- #
def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    # ------------------------------------------------------------------ #
    # 1) Non-stationary scenarios (the new Requirement 3 environment).
    # ------------------------------------------------------------------ #
    ns_scenarios = [
        NonStationaryMultiCampaignScenario.load(p)
        for p in sorted(SCENARIO_DIR.glob("ns_*.json"))
    ]
    for scenario in ns_scenarios:
        is_binding = scenario.name.endswith("_binding")
        sequence = scenario.make_sequence()
        opt = hindsight_multi_campaign_opt(
            scenario.valuation_array(), scenario.bid_array(), sequence,
            scenario.rho, scenario.edge_list(),
        )
        dyn_total = dynamic_total_opt(
            scenario.valuation_array(), scenario.bid_array(), sequence,
            scenario.rho, scenario.edge_list(),
        )
        make_env = lambda seed, s=scenario: NonStationaryMultiCampaignEnv(s)

        pd_runs = run_pd(scenario, make_env)
        pd_regret = scenario.horizon * opt.value - pd_runs["mean_total_util"]
        cucb_runs = None
        cucb_regret = float("nan")
        if is_binding:
            cucb_runs = run_cucb(scenario, make_env)
            cucb_regret = scenario.horizon * opt.value - cucb_runs["mean_total_util"]
            plot_regret(scenario, opt.value, pd_runs, cucb_runs, scenario.schedule["kind"])
            plot_budget(scenario, pd_runs)
            plot_dual(scenario, pd_runs)

        print(
            f"[{scenario.name}] kind={scenario.schedule['kind']} rho={scenario.rho:.3f} "
            f"T*OPT_fixed={scenario.horizon * opt.value:.0f} dyn={dyn_total:.0f} | "
            f"PD util={pd_runs['mean_total_util']:.0f} (reg {pd_regret:.0f}) "
            f"CUCB reg={cucb_regret:.0f}"
        )
        rows.append({
            "scenario": scenario.name,
            "world": "nonstationary",
            "kind": scenario.schedule["kind"],
            "n_campaigns": scenario.n_campaigns,
            "rho": round(scenario.rho, 4),
            "opt_per_round": round(opt.value, 4),
            "T_OPT_fixed": round(scenario.horizon * opt.value, 1),
            "T_OPT_dynamic": round(dyn_total, 1),
            "pd_total_util": round(pd_runs["mean_total_util"], 1),
            "pd_regret": round(pd_regret, 1),
            "cucb_regret": round(cucb_regret, 1),
            "pd_total_cost": round(pd_runs["mean_total_cost"], 1),
            "budget": round(scenario.budget, 1),
            "pd_stop_round": round(pd_runs["mean_stop"], 0),
        })

    # ------------------------------------------------------------------ #
    # 2) Stochastic scenarios (reuse Requirement 2 data) -> the other world.
    # ------------------------------------------------------------------ #
    stoch_scenarios = [
        MultiCampaignScenario.load(p)
        for p in sorted(SCENARIO_DIR.glob("multi_*_binding.json"))
    ]
    for scenario in stoch_scenarios:
        true_opt = multi_campaign_opt(scenario)
        make_env = lambda seed, s=scenario: MultiCampaignStochasticEnv(
            s, np.random.default_rng(2000 + seed)
        )
        pd_runs = run_pd(scenario, make_env)
        cucb_runs = run_cucb(scenario, make_env)
        pd_regret = scenario.horizon * true_opt.value - pd_runs["mean_total_util"]
        cucb_regret = scenario.horizon * true_opt.value - cucb_runs["mean_total_util"]
        plot_regret(scenario, true_opt.value, pd_runs, cucb_runs, "stochastic")

        print(
            f"[{scenario.name}] stochastic rho={scenario.rho:.3f} "
            f"T*OPT={scenario.horizon * true_opt.value:.0f} | "
            f"PD util={pd_runs['mean_total_util']:.0f} (reg {pd_regret:.0f}) "
            f"CUCB reg={cucb_regret:.0f}"
        )
        rows.append({
            "scenario": scenario.name,
            "world": "stochastic",
            "kind": "stationary",
            "n_campaigns": scenario.n_campaigns,
            "rho": round(scenario.rho, 4),
            "opt_per_round": round(true_opt.value, 4),
            "T_OPT_fixed": round(scenario.horizon * true_opt.value, 1),
            "T_OPT_dynamic": round(scenario.horizon * true_opt.value, 1),
            "pd_total_util": round(pd_runs["mean_total_util"], 1),
            "pd_regret": round(pd_regret, 1),
            "cucb_regret": round(cucb_regret, 1),
            "pd_total_cost": round(pd_runs["mean_total_cost"], 1),
            "budget": round(scenario.budget, 1),
            "pd_stop_round": round(pd_runs["mean_stop"], 0),
        })

    # ------------------------------------------------------------------ #
    # 3) Best-of-both-worlds summary + regret scaling.
    # ------------------------------------------------------------------ #
    summary_rows = [r for r in rows if not np.isnan(r["cucb_regret"])]
    plot_bobw_summary(summary_rows)

    print("\nScaling experiment (PD regret vs T) on ns_fast_binding ...")
    base = next(s for s in ns_scenarios if s.name == "ns_fast_binding")
    plot_scaling(base)

    with SUMMARY_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved plots to {PLOT_DIR} and summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
