"""Requirement 3: best-of-both-worlds primal-dual bidding for many campaigns.

This is a single algorithm that attains sublinear regret in **both** the
stochastic environment of Requirement 2 *and* a highly non-stationary
(adversarial) environment, against the best fixed feasible bidding distribution
in hindsight.  It follows the primal-dual recipe for online optimization with a
long-term (budget) constraint (slide 16 of ``8-GeneralAuctions.pdf``; Castiglioni
et al., 2022):

* a single **dual** variable ``lambda >= 0`` prices the global per-round budget
  ``rho = B / T``; it is updated by projected online gradient descent on the
  budget violation;
* given ``lambda``, the per-round problem becomes an *unconstrained* combinatorial
  bid-selection on the **Lagrangian reward**
  ``L_{i,t}(b) = f_{i,t}(b) - lambda_t * c_{i,t}(b)``, which is handed to a
  **primal regret minimizer**.

**Primal regret minimizer for this specific problem.**  Requirement 3 grants
*full feedback*: after each round we observe every campaign's highest competing
bid ``m_i``, hence the Lagrangian reward of *every* (campaign, bid) pair, played
or not.  The natural full-information regret minimizer is therefore **Hedge**
(multiplicative weights), which -- unlike UCB -- never stops "exploring" and is
robust to arbitrarily changing rewards, giving the best-of-both-worlds behaviour.

The feasible joint actions are the conflict-graph independent sets together with
a bid per chosen campaign.  For the project's conflict graphs (a *matching*:
disjoint conflict pairs) this set **factorizes** into independent blocks:

* a *free* campaign ``i`` -> actions ``{skip} U {(i, b) : b in B}``;
* a conflict pair ``(i, j)`` -> actions ``{skip} U {(i, b)} U {(j, b)}``
  (mutually exclusive, exactly encoding the edge constraint).

Running an independent Hedge instance per block is a regret minimizer over the
whole feasible set (the product of the per-block simplices), and the single dual
variable ``lambda`` couples the blocks through the shared budget price.  This is
the "primal regret minimizer designed for the specific problem under study".

Per round ``t``:

1. each block draws an action from its Hedge distribution -> a conflict-feasible
   joint bid profile (an independent set + a bid per played campaign);
2. play it, observe the full ``m`` vector (full feedback);
3. feed every block the Lagrangian reward of *all* its actions at ``lambda_t``;
4. dual step ``lambda <- proj_{[0, 1/rho]}( lambda + eta_dual (c_t - rho) )``
   where ``c_t`` is the (expected, under the primal mix) per-round spend.

Budget accounting / the "stop when you cannot afford a round" rule live in
:mod:`ola.simulation`, exactly as for the UCB bidders.
"""

from __future__ import annotations

import numpy as np


class _HedgeBlock:
    """A full-information Hedge instance over one block's mutually exclusive actions.

    Action ``0`` is always *skip* (play nothing in this block, paying nothing);
    the remaining actions are the ``(campaign, bid)`` pairs of the block, stored
    column-wise in ``camp_ids`` / ``bid_ids`` for vectorized scoring.  Weights are
    ``w(a) ~ exp(eta * G(a))`` with ``G`` the cumulative Lagrangian reward.
    """

    def __init__(self, camp_ids: np.ndarray, bid_ids: np.ndarray, eta: float):
        self.camp_ids = np.asarray(camp_ids, dtype=np.int64)  # playable actions
        self.bid_ids = np.asarray(bid_ids, dtype=np.int64)
        self.n_actions = 1 + self.camp_ids.size  # +1 for the skip action (index 0)
        self.eta = float(eta)
        self.cum = np.zeros(self.n_actions, dtype=float)

    def reset(self) -> None:
        self.cum[:] = 0.0

    def distribution(self) -> np.ndarray:
        z = self.eta * self.cum
        z -= z.max()  # log-sum-exp stabilization
        w = np.exp(z)
        return w / w.sum()

    def update(self, reward: np.ndarray) -> None:
        self.cum += reward


class PrimalDualBidder:
    """Best-of-both-worlds primal-dual bidder (multi-campaign, full feedback)."""

    def __init__(
        self,
        valuations: np.ndarray,
        bids: np.ndarray,
        horizon: int,
        rho: float,
        conflict_edges: list[tuple[int, int]],
        rng: np.random.Generator,
        eta_primal: float | None = None,
        eta_dual: float | None = None,
        lambda_max: float | None = None,
    ):
        self.valuations = np.asarray(valuations, dtype=float)
        self.bids = np.asarray(bids, dtype=float)
        self.n_campaigns = self.valuations.size
        self.n_bids = self.bids.size
        self.horizon = int(horizon)
        self.rho = float(rho)
        self.conflict_edges = [(int(i), int(j)) for i, j in conflict_edges]
        self.rng = rng

        # The dual variable lives in [0, lambda_max]; 1/rho is the standard bound
        # (beyond it skipping every campaign already dominates any positive bid).
        self.lambda_max = float(lambda_max) if lambda_max is not None else 1.0 / self.rho

        # Range of the per-round Lagrangian reward of a single (campaign, bid):
        # utility in [0, max v], penalized cost in [0, lambda_max * max bid].
        max_bid = float(self.bids.max())
        self.reward_range = float(self.valuations.max() + self.lambda_max * max_bid)

        # Worst-case per-round spend bounds the dual gradient (c_t - rho).
        edge_nodes = {n for e in self.conflict_edges for n in e}
        self._free_nodes = [i for i in range(self.n_campaigns) if i not in edge_nodes]
        max_active = len(self._free_nodes) + len(self.conflict_edges)
        self._grad_bound = max(max_active * max_bid, self.rho)

        T = max(self.horizon, 2)
        self._eta_primal_arg = eta_primal
        self.eta_dual = (
            float(eta_dual)
            if eta_dual is not None
            else self.lambda_max / (self._grad_bound * np.sqrt(T))
        )

        self._build_blocks()
        self.reset()

    # ------------------------------------------------------------------ #
    def _build_blocks(self) -> None:
        """Decompose the feasible set into per-block mutually exclusive actions."""
        T = max(self.horizon, 2)
        all_bids = np.arange(self.n_bids, dtype=np.int64)

        def make_block(camps: list[int]) -> _HedgeBlock:
            # Playable actions: every (campaign, bid) for the campaigns in the block.
            camp_ids = np.repeat(np.asarray(camps, dtype=np.int64), self.n_bids)
            bid_ids = np.tile(all_bids, len(camps))
            n_actions = 1 + camp_ids.size
            if self._eta_primal_arg is not None:
                eta = float(self._eta_primal_arg)
            else:
                k = max(n_actions, 2)
                eta = np.sqrt(8.0 * np.log(k) / T) / self.reward_range
            return _HedgeBlock(camp_ids, bid_ids, eta)

        self.blocks: list[_HedgeBlock] = []
        for i, j in self.conflict_edges:
            self.blocks.append(make_block([i, j]))
        for i in self._free_nodes:
            self.blocks.append(make_block([i]))

    def reset(self) -> None:
        for block in self.blocks:
            block.reset()
        self.lam = 0.0
        self.t = 0
        self.lambda_history = np.zeros(self.horizon)
        self._last_dists: list[np.ndarray] = []

    # ------------------------------------------------------------------ #
    def select_superarm(self) -> dict[int, int]:
        """Sample one conflict-feasible joint action from the per-block Hedge."""
        actions: dict[int, int] = {}
        self._last_dists = []
        for block in self.blocks:
            p = block.distribution()
            self._last_dists.append(p)
            k = int(self.rng.choice(block.n_actions, p=p))
            if k > 0:  # 0 is the skip action
                campaign = int(block.camp_ids[k - 1])
                bid_index = int(block.bid_ids[k - 1])
                actions[campaign] = bid_index
        return actions

    def update(
        self,
        per_campaign: dict[int, tuple[int, bool, float, float]],
        competing_bids: np.ndarray,
    ) -> None:
        """Full-feedback update of the primal blocks and the dual variable.

        ``competing_bids`` is the round's whole ``m`` vector (all campaigns), as
        Requirement 3 assumes full feedback.  ``per_campaign`` (the played arms)
        is unused for learning but kept for interface symmetry with the UCB
        bidders.
        """
        m = np.asarray(competing_bids, dtype=float)
        lam = self.lam

        # Per (campaign, bid) Lagrangian reward under the current dual price.
        win = self.bids[None, :] >= m[:, None]                       # (N, B)
        f = (self.valuations[:, None] - self.bids[None, :]) * win    # (N, B)
        c = (self.bids[None, :] * win)                              # (N, B)
        lagrangian = f - lam * c                                    # (N, B)

        # Feed each block the Lagrangian reward of all its actions (skip -> 0),
        # and accumulate the expected per-round spend of the primal mixture.
        expected_cost = 0.0
        for block, p in zip(self.blocks, self._last_dists):
            reward = np.empty(block.n_actions)
            cost_vec = np.empty(block.n_actions)
            reward[0] = 0.0  # skip action
            cost_vec[0] = 0.0
            reward[1:] = lagrangian[block.camp_ids, block.bid_ids]
            cost_vec[1:] = c[block.camp_ids, block.bid_ids]
            block.update(reward)
            expected_cost += float(p @ cost_vec)

        # Dual projected online gradient descent on the budget violation.
        self.lam = float(
            np.clip(lam + self.eta_dual * (expected_cost - self.rho), 0.0, self.lambda_max)
        )
        if self.t < self.horizon:
            self.lambda_history[self.t] = self.lam
        self.t += 1

    @property
    def name(self) -> str:
        return type(self).__name__
