"""Online Learning Applications - bidding under budget constraints.

This package implements the agency bidding project for the OLA course.
Requirement 1 (single campaign, stochastic environment, UCB1 with and without
budget), Requirement 2 (multiple campaigns with a global budget and conflict
graph, Combinatorial-UCB) and Requirement 3 (best-of-both-worlds primal-dual
bidding for stochastic *and* highly non-stationary environments, with full
feedback) are fully implemented.
"""

__all__ = [
    "distributions",
    "auction",
    "environment",
    "baseline",
    "metrics",
    "simulation",
    "conflict_graph",
    "algorithms",
]
