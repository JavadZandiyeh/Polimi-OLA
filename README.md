# OLA Project - Bidding under Budget Constraints

Online learning algorithms for an advertising agency that bids dynamically on
multiple first-price-auction campaigns under budget constraints.

This repository implements **Requirement 1** (single campaign, stochastic
environment), **Requirement 2** (multiple campaigns, stochastic environment),
**Requirement 3** (best-of-both-worlds: a single primal-dual algorithm for
multiple campaigns that works in *both* the stochastic and a highly
non-stationary environment, with full feedback) and **Requirement 4** (slightly /
piecewise-stationary multiple campaigns: Combinatorial-UCB with a sliding window
and with a CUSUM change detector, compared against the primal-dual method).

## Setting (from the course material)

For a single campaign and a first-price auction run at every round
`t = 1, ..., T`:

1. The advertiser picks a bid `b_t` from a small discrete set `B`.
2. The (stochastic) environment draws the highest competing bid `m_t ~ D`.
3. The advertiser wins iff `b_t >= m_t`.
4. On a win: utility `f_t(b_t) = v - b_t` and cost `c_t(b_t) = b_t`
   (first price -> you pay your own bid). On a loss both are `0`.
5. The budget `B` is decreased by the cost. When the remaining budget can no
   longer cover the largest bid, the advertiser stops participating.

Feedback is **bandit**: the advertiser only observes the utility/cost of the
bid it actually played, not `m_t` and not the counterfactual reward of other
bids.

## Algorithms (Requirement 1)

- **Algorithm A - `UCB1Bidder`** (`src/ola/algorithms/ucb1.py`): each bid is an
  arm, the reward is the auction utility `f_t(b) in [0, v]`. Classic UCB1 index
  `mean(b) + sqrt(2 log t / N(b))` (rescaled to the reward range). It **ignores
  the budget constraint** when choosing bids. Baseline for regret: the best
  fixed bid in hindsight, `T * max_b E[f(b)]`.

- **Algorithm B - `BudgetedUCBBidder`** (`src/ola/algorithms/ucb_budget.py`):
  the UCB-bidding / "bandits with knapsacks" approach (Agrawal & Devanur 2014,
  slide 18 of `8-GeneralAuctions.pdf`). It keeps an upper confidence bound on
  the utility `f_UCB(b)` and a lower confidence bound on the cost `c_LCB(b)`,
  and every round solves the linear program

  ```
  max_gamma   sum_b gamma(b) f_UCB(b)
  s.t.        sum_b gamma(b) c_LCB(b) <= rho      (rho = B / T)
              gamma in simplex
  ```

  then samples `b_t ~ gamma_t`. Baseline for regret: the constrained
  clairvoyant `T * OPT`, with `OPT = sup_gamma f_bar(gamma)` s.t.
  `c_bar(gamma) <= rho` computed from the true distribution.

## Multiple campaigns (Requirement 2)

Now the agency runs `N` campaigns at once. Each round a **joint** vector of
highest competing bids `m = (m_1, ..., m_N)` is drawn (campaigns may be
correlated via a Gaussian copula), the agency picks a bid per campaign, wins
campaign `i` iff `b_i >= m_i`, and pays from a single **global** budget. Two
extra ingredients matter:

- a **conflict graph**: non-compatible campaigns (e.g. Coca-Cola/Pepsi) cannot
  be run in the same round, so the played campaigns must form an independent set;
- **semi-bandit feedback**: only the played campaigns' outcomes are observed.

- **`CombinatorialUCBBidder`** (`src/ola/algorithms/combinatorial_ucb.py`) is
  Combinatorial-UCB with a budget constraint (Part 9 of the course), the
  combinatorial extension of Algorithm B. Each `(campaign, bid)` is an arm with
  an optimistic utility bound `f_UCB(i,b)` and pessimistic cost bound
  `c_LCB(i,b)`. The per-round optimization oracle solves the LP

  ```
  max_x   sum_{i,b} f_UCB(i,b) x(i,b)
  s.t.    sum_b x(i,b) <= 1                          (each campaign at most once)
          sum_{i,b} c_LCB(i,b) x(i,b) <= rho         (global per-round budget)
          sum_b x(i,b) + sum_b x(j,b) <= 1           (conflict edge (i,j))
          x >= 0
  ```

  and the fractional solution is rounded into a feasible joint action with
  conflict-aware sampling (independent campaigns Bernoulli-rounded; each conflict
  pair rounded mutually exclusively, which is exact for the matching-type
  conflict graphs used here). Baseline for regret: the combinatorial clairvoyant
  `T * OPT` from the same LP with the true means.

## Best-of-both-worlds with multiple campaigns (Requirement 3)

Requirement 3 keeps the multi-campaign setting but adds a **highly
non-stationary** environment: the highest competing bid of each campaign follows
a *non-stochastic* sequence whose distribution changes over time. The goal is a
**single** algorithm that has sublinear regret in **both** the stochastic
environment of req2 *and* the non-stationary one — the "best of both worlds".
Here the agency has **full feedback**: every round it observes the highest
competing bid `m_i` of *every* campaign (played or not).

- **`PrimalDualBidder`** (`src/ola/algorithms/primal_dual.py`) is a **primal-dual**
  strategy with a single global budget constraint (slide 16 of
  `8-GeneralAuctions.pdf`; Castiglioni et al., 2022):

  - a single **dual** variable `lambda >= 0` prices the per-round budget
    `rho = B / T` and is updated by projected online gradient descent on the
    budget violation, `lambda <- proj_[0,1/rho]( lambda + eta (c_t - rho) )`;
  - given `lambda`, the per-round problem reduces to maximizing the **Lagrangian
    reward** `L_{i,t}(b) = f_i(b) - lambda c_i(b)`, handed to a **primal regret
    minimizer**.

  The **primal regret minimizer designed for this problem** exploits full
  feedback (every `(campaign, bid)` Lagrangian reward is observable) and uses
  **Hedge** (multiplicative weights) — which, unlike UCB, never stops exploring
  and is robust to arbitrarily changing rewards. The feasible joint actions
  (conflict-graph independent sets + a bid per campaign) **factorize** over the
  matching conflict graph into independent **blocks** (a free campaign, or a
  mutually-exclusive conflict pair); one Hedge instance per block is a regret
  minimizer over the whole feasible set, and the shared `lambda` couples the
  blocks through the budget. Regret is measured against the **best fixed feasible
  bidding distribution in hindsight** (`hindsight_multi_campaign_opt`), which
  reduces to the req2 clairvoyant in the stochastic limit.

## Slightly non-stationary with multiple campaigns (Requirement 4)

Requirement 4 studies a **piecewise-stationary** multi-campaign environment: the
horizon is split into intervals, and within each interval the joint distribution
of highest competing bids is fixed but *different* from the neighbouring
intervals (the "abrupt changes" of `10-nonStationary.pdf`). The req2
Combinatorial-UCB fails here — it shrinks its confidence bounds and "converges",
so it reacts slowly to interval boundaries and tends to deplete its budget early.
Two standard fixes (Part 10 of the course) are lifted to the combinatorial,
budget-constrained setting (`src/ola/algorithms/nonstationary_ucb.py`):

- **`SlidingWindowCombinatorialUCBBidder`** — *passive* forgetting: every
  `(campaign, bid)` arm is estimated from only the last `W` rounds, so the
  confidence bounds never collapse. `W ~ sqrt(T)` is the textbook rule; here it is
  enlarged (`~20 sqrt(T)`) because the combinatorial action spreads plays over
  many arms, and the best `W` tracks the (unknown) interval length.
- **`ChangeDetectionCombinatorialUCBBidder`** — *active* detection: a per-arm
  **CUSUM** detector monitors each arm's utility and, on a flagged change, resets
  that arm (it becomes fresh/optimistic and is re-estimated); a small uniform
  exploration probability keeps feeding samples to every arm.

Both reuse req2's per-round budget LP oracle and conflict-aware rounding and keep
semi-bandit feedback. They are compared against each other and the req3
**`PrimalDualBidder`** on the piecewise-stationary scenarios, with regret measured
against the **dynamic per-interval optimum** (`piecewise_dynamic_opt`).

## Layout

```
code/
  requirements.txt
  src/ola/
    distributions.py        competing-bid distributions (sample/cdf/ppf) + joint
    auction.py              first-price utility/cost/win mechanics
    environment.py          single-/multi-campaign stochastic + non-stationary envs
    baseline.py             clairvoyant + hindsight + per-interval dynamic baselines
    metrics.py              regret / budget bookkeeping
    simulation.py           run single-/multi-campaign and primal-dual episodes
    conflict_graph.py       conflict graph (independent-set queries, matching)
    algorithms/
      ucb1.py               Algorithm A (req1)
      ucb_budget.py         Algorithm B (req1)
      combinatorial_ucb.py  Combinatorial-UCB with budget (req2)
      primal_dual.py        best-of-both-worlds primal-dual bidder (req3)
      nonstationary_ucb.py  sliding-window + CUSUM Combinatorial-UCB (req4)
      pacing.py             single-campaign primal-dual pacing (req3 special case)
  data/
    generate_datasets.py    synthesize and persist all scenarios
    scenarios/*.json        scenario configs (single-, multi-, non-stationary)
    samples/*.csv           precomputed competing-bid sequences (reproducible)
  experiments/
    run_requirement1.py     req1: UCB1 vs budgeted UCB, regret + budget plots
    run_requirement2.py     req2: Combinatorial-UCB, regret + budget plots
    run_requirement3.py     req3: primal-dual (stochastic + non-stationary) vs Comb-UCB
    run_requirement4.py     req4: SW-UCB vs CD-UCB vs primal-dual (piecewise-stationary)
    plots/                  output figures
```

## How to run

```bash
cd code
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# (re)generate the synthesized datasets
./.venv/bin/python data/generate_datasets.py

# run Requirement 1 experiments (regret curves + budget plots)
./.venv/bin/python experiments/run_requirement1.py

# run Requirement 2 experiments (multi-campaign Combinatorial-UCB)
./.venv/bin/python experiments/run_requirement2.py

# run Requirement 3 experiments (best-of-both-worlds primal-dual)
./.venv/bin/python experiments/run_requirement3.py

# run Requirement 4 experiments (sliding window vs change detection vs primal-dual)
./.venv/bin/python experiments/run_requirement4.py
```

Figures are written to `experiments/plots/`.

## Synthesized datasets

The "dataset" of a stochastic environment is the distribution of the highest
competing bid. `data/generate_datasets.py` produces:

- **single-campaign** scenarios (uniform, low/high competition truncated normals,
  skewed Beta, bimodal mixture), each in a budget-binding and a slack variant
  (used by Requirement 1);
- **multi-campaign** scenarios (used by Requirement 2), each in a binding and a
  slack variant:
  - `multi_independent` (3 independent campaigns, no conflicts),
  - `multi_conflict` (4 campaigns with conflict edges `{0-1, 2-3}`),
  - `multi_correlated` (3 campaigns with positively correlated competing bids,
    Gaussian copula `rho = 0.6`).

- **non-stationary** scenarios (used by Requirement 3), each in a binding and a
  slack variant. Their "dataset" is a *fixed, non-stochastic* `(T, N)` sequence
  of competing bids whose underlying distribution changes over time, following
  the course taxonomy of non-stationary environments (`10-nonStationary.pdf`):
  - `ns_abrupt` (3 campaigns, piecewise-stationary, 5 phases — a competitor
    enters/leaves so the best bid jumps),
  - `ns_smooth` (3 campaigns, sinusoidally drifting competition),
  - `ns_fast` (3 campaigns, a fresh distribution every 50 rounds — the highly
    non-stationary regime emphasized by req3),
  - `ns_conflict` (4 campaigns with conflict edges `{0-1, 2-3}` and abrupt phases
    that flip which campaign of each pair is the better buy, forcing switches).
- **piecewise-stationary** scenarios for Requirement 4 (`abrupt` schedule with a
  controlled number of intervals): `pw_slight` (3 long intervals, "slightly"
  non-stationary) and `pw_frequent` (8 shorter intervals); together with the
  reused `ns_abrupt` (5 intervals) and `ns_conflict` (4 intervals + conflict
  graph) they span a range of change frequencies.

Each scenario stores a JSON config in `data/scenarios/` and a reproducible
competing-bid sequence in `data/samples/` (a vector per round for multi-campaign
and non-stationary scenarios). Budget regimes are derived from each instance's
unconstrained optimal spend (the realized-sequence hindsight spend for the
non-stationary instances) so the binding variants genuinely activate the
constraint.

## Results (Requirement 1)

Running `experiments/run_requirement1.py` (5 seeds, `T = 10000`) produces the
figures in `experiments/plots/` and `experiments/summary.csv`. Headline numbers
on the budget-binding scenarios (total utility; higher is better):

| scenario                 | rho    | UCB1 util (stop) | Budgeted util (stop) | T*OPT |
|--------------------------|--------|------------------|----------------------|-------|
| uniform_binding          | 0.080  | 346  (t~3325)    | 1021 (t~7872)        | 1455  |
| low_competition_binding  | 0.163  | 1631 (t~4812)    | 3077 (t~8353)        | 3672  |
| beta_skewed_binding      | 0.153  | 1109 (t~4678)    | 2154 (t~7931)        | 2779  |
| bimodal_binding          | 0.083  | 633  (t~3140)    | 1473 (t~6036)        | 2286  |

Takeaways:

* **Ignoring the budget is costly.** UCB1 converges to the unconstrained best
  bid, which spends well above `rho` per round, so it exhausts the budget early
  and then earns nothing. In `regret_*_binding.png` its regret is flat
  (sublinear) until depletion and then kinks to a straight line.
* **Budget-aware bidding paces.** Budgeted UCB tracks the ideal pacing line
  `B - rho t` (see `budget_*_binding.png`), lasts far longer and roughly doubles
  the utility. Its learned bid distribution concentrates on the constrained
  `OPT` mixture (see `diagnostics_*_binding.png`).
* **Sublinear regret.** With a slack budget (`*_slack`) both algorithms approach
  `T*OPT`, confirming UCB1 is a valid no-regret learner when the budget does not
  bind. `scaling_regret.png` shows the budgeted algorithm's final regret growing
  like `sqrt(T)` across `T in {2k, ..., 40k}`, matching the
  `O~(sqrt(T))` guarantee (Agrawal & Devanur, 2014).

## Results (Requirement 2)

Running `experiments/run_requirement2.py` (5 seeds, `T = 10000`) produces the
multi-campaign figures and `experiments/summary_req2.csv`. Headline numbers
(total utility; higher is better):

| scenario                   | N | corr | rho   | Comb-UCB util (stop) | total cost / budget | T*OPT |
|----------------------------|---|------|-------|----------------------|---------------------|-------|
| multi_independent_binding  | 3 | 0.0  | 0.503 | 5653 (t~10000)       | 4759 / 5027         | 6350  |
| multi_conflict_binding     | 4 | 0.0  | 0.353 | 4316 (t~9025)        | 3526 / 3527         | 5302  |
| multi_correlated_binding   | 3 | 0.6  | 0.316 | 4568 (t~7025)        | 3162 / 3164         | 6713  |

Takeaways:

* **Combinatorial-UCB learns the joint bid profile.** Per-campaign learned bid
  frequencies concentrate on the clairvoyant `OPT` allocation
  (`diagnostics_multi_*_binding.png`), and the cumulative regret against the
  combinatorial `T*OPT` is sublinear (`regret_multi_*.png`), flattening out
  before the budget is depleted.
* **The global budget is respected and paced.** Realized spend stays within the
  budget and tracks the ideal pacing line `B - rho t`
  (`budget_multi_*_binding.png`).
* **Conflicts are handled by the oracle.** With the conflict graph, the agency
  consistently drops the lower-value campaign of each conflicting pair (e.g.
  campaign 3 stays idle in favour of the higher-value campaign 2), exactly as the
  clairvoyant `OPT` prescribes.
* **Sublinear scaling.** `scaling_regret_multi.png` shows final regret growing
  like `sqrt(T)` across `T in {2k, ..., 40k}`, matching the `O~(sqrt(T))`
  Combinatorial-UCB-with-budget guarantee.

## Results (Requirement 3)

Running `experiments/run_requirement3.py` (`T = 10000`, 5 seeds for primal-dual,
3 for Comb-UCB) runs the *same* `PrimalDualBidder` on both the stochastic req2
scenarios and the non-stationary `ns_*` scenarios, compares it to the
stochastic-only Combinatorial-UCB, and writes `experiments/summary_req3.csv` plus
the figures below. Regret is against the best fixed feasible bidding distribution
in hindsight (`T*OPT_fixed`). Final-regret headline (lower is better):

| scenario                  | world         | T*OPT | Primal-Dual regret | Comb-UCB regret |
|---------------------------|---------------|-------|--------------------|-----------------|
| ns_abrupt_binding         | non-stationary| 5677  | 690                | 222             |
| ns_smooth_binding         | non-stationary| 5328  | 1456               | 1910            |
| ns_fast_binding           | non-stationary| 5511  | 512                | 1173            |
| ns_conflict_binding       | non-stationary| 3589  | 439                | 522             |
| multi_independent_binding | stochastic    | 6350  | 431                | 700             |
| multi_conflict_binding    | stochastic    | 5302  | 405                | 1024            |
| multi_correlated_binding  | stochastic    | 6713  | 734                | 2154            |

Takeaways:

* **One algorithm, both worlds.** The primal-dual bidder keeps regret a small
  fraction of `T*OPT` in *every* regime — stochastic and non-stationary alike
  (`bobw_summary.png`, `regret_*.png`). This is the best-of-both-worlds property.
* **The stochastic-only algorithm is not robust.** Combinatorial-UCB (UCB-style)
  reduces exploration and paces with stale estimates, so under non-stationarity
  (and even on the stochastic binding scenarios) it tends to **deplete its budget
  early** and then earn nothing — its regret stays flat and then **kinks into a
  straight line** (`regret_ns_fast_binding.png`, `regret_multi_correlated_binding.png`),
  the failure mode of stochastic algorithms in changing environments
  (`10-nonStationary.pdf`). Primal-dual paces all the way to `T`.
* **Primal-dual paces via the dual.** Remaining budget tracks the ideal line
  `B - rho t` (`budget_*.png`) and the dual variable `lambda_t` rises when the
  agency overspends and falls when it underspends (`dual_*.png`), reacting in
  real time to the (fast) changes in competition.
* **Price of non-stationarity.** `summary_req3.csv` also reports a *dynamic*
  (piecewise) hindsight optimum `T*OPT_dynamic` that may track the changes; the
  gap to `T*OPT_fixed` measures how non-stationary each instance is.
* **Sublinear scaling.** `scaling_regret_pd.png` shows the primal-dual final
  regret growing like `sqrt(T)` across `T in {2k, ..., 40k}` on `ns_fast`.

## Results (Requirement 4)

Running `experiments/run_requirement4.py` (`T = 10000`, 4 seeds) compares
Comb-UCB, SW-CombUCB, CD-CombUCB and Primal-Dual on the piecewise-stationary
scenarios and writes `experiments/summary_req4.csv` plus the figures below.
Regret is against the dynamic per-interval optimum `T*OPT_dyn` (lower is better):

| scenario            | intervals | T*OPT_dyn | Comb-UCB | SW-CombUCB | CD-CombUCB | Primal-Dual |
|---------------------|-----------|-----------|----------|------------|------------|-------------|
| pw_slight_binding   | 3         | 6823      | 1622     | **944**    | 1805       | 1051        |
| ns_conflict_binding | 4         | 4660      | 1577     | 1553       | 1688       | **1513**    |
| ns_abrupt_binding   | 5         | 6267      | **868**  | 1351       | 1071       | 1281        |
| pw_frequent_binding | 8         | 5413      | **748**  | 784        | 1489       | 986         |

Takeaways:

* **No single winner — the comparison is the point.** Which method wins depends on
  how non-stationary and how budget-tight the instance is. The sliding window
  wins when the budget binds and intervals are long (`pw_slight`); plain Comb-UCB
  is hard to beat when the changes are mild/late (`ns_abrupt`, `pw_frequent`),
  matching the course remark that non-stationary methods pay an extra price in
  almost-stationary regimes.
* **Budget pacing is where primal-dual shines.** In the tight-budget conflict
  scenario the three UCB variants **exhaust their budget by round ~6000** and earn
  nothing afterwards (their regret kinks into a straight line), whereas
  Primal-Dual tracks the ideal pacing line `B - rho t` to the horizon
  (`budget_req4_ns_conflict_binding.png`) and has the lowest regret. The same
  early-depletion kink is visible for Comb-UCB/CD on `pw_slight`
  (`regret_req4_pw_slight_binding.png`).
* **Change detection is sensitive.** CUSUM helps only when changes are infrequent
  enough to estimate between them; with frequent changes (`pw_frequent`) the
  resets and exploration waste the binding budget and it is the worst method.
* **The window size matters.** `window_sensitivity.png` shows a clear U-shape: a
  small `W` (the plain `sqrt(T)` rule) badly under-samples the combinatorial arms,
  while the regret is minimized when `W` is on the order of the interval length —
  i.e. the best `W` depends on the unknown change frequency, exactly as the course
  notes.

## Notes

`AdversarialSingleCampaignEnv` in `environment.py` remains a documented stub for a
single-campaign adversarial variant; the single-campaign primal-dual special case
is available as `PacingBidder` (`pacing.py`), a thin wrapper around the
multi-campaign `PrimalDualBidder`.
