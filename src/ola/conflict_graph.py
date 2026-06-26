"""Conflict graph between campaigns (multi-campaign requirement).

The agency cannot run two non-compatible campaigns (e.g. Coca-Cola and Pepsi)
in the same round.  This is modeled by a conflict graph: nodes are campaigns and
an edge means the two campaigns are mutually exclusive.  At each round the set of
campaigns bid on must therefore be an **independent set** of this graph.

In Requirement 2 the *selection* of which campaigns to run (and the bid in each)
is folded into the per-round LP oracle in :mod:`ola.baseline` via edge
constraints, and the conflict-aware rounding lives in
:mod:`ola.algorithms.combinatorial_ucb`.  This module provides the conflict
graph data structure plus the validation/queries those components rely on.
"""

from __future__ import annotations

from typing import Iterable

import networkx as nx


class ConflictGraph:
    """Undirected graph of mutually exclusive campaigns."""

    def __init__(self, n_campaigns: int, edges: Iterable[tuple[int, int]] = ()):
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(n_campaigns))
        self.graph.add_edges_from((int(u), int(v)) for u, v in edges)

    @property
    def n_campaigns(self) -> int:
        return self.graph.number_of_nodes()

    def edges(self) -> list[tuple[int, int]]:
        return [(int(u), int(v)) for u, v in self.graph.edges()]

    def are_compatible(self, i: int, j: int) -> bool:
        """True if campaigns ``i`` and ``j`` can run in the same round."""
        return not self.graph.has_edge(i, j)

    def is_independent_set(self, campaigns: Iterable[int]) -> bool:
        """True if no two campaigns in the set conflict."""
        campaigns = list(campaigns)
        for a in range(len(campaigns)):
            for b in range(a + 1, len(campaigns)):
                if self.graph.has_edge(campaigns[a], campaigns[b]):
                    return False
        return True

    def is_matching(self) -> bool:
        """True if every node has degree <= 1 (the conflict graph is a matching).

        Matchings (disjoint conflict pairs) are exactly the case where the
        edge-constraint LP relaxation is tight and the per-edge rounding in the
        Combinatorial-UCB sampler is exact.
        """
        return all(deg <= 1 for _, deg in self.graph.degree())

    def maximal_independent_set(self, seed: int | None = None) -> list[int]:
        """A (greedy) maximal independent set -- a feasible set of campaigns."""
        return sorted(nx.maximal_independent_set(self.graph, seed=seed))

    def max_independent_set_size(self) -> int:
        """Size of a maximum independent set (used to bound the per-round cost).

        Computed exactly for a matching (``n_nodes - n_edges``); for a general
        graph it falls back to the complement's clique number via ``networkx``.
        """
        if self.is_matching():
            return self.graph.number_of_nodes() - self.graph.number_of_edges()
        complement = nx.complement(self.graph)
        return nx.graph_clique_number(complement)
