"""Shared utility helpers for graph normalization and safe casting."""

import networkx as nx
import numpy as np


def as_simple_undirected(G: nx.Graph) -> nx.Graph:
    """Привести граф к простому неориентированному виду.

    Предполагается, что weight/confidence уже числовые (это обязан гарантировать preprocess).
    """
    H = G
    if hasattr(H, "is_directed") and H.is_directed():
        H = H.to_undirected(as_view=False)

    if isinstance(H, (nx.MultiGraph, nx.MultiDiGraph)):
        simple = nx.Graph()
        simple.add_nodes_from(H.nodes(data=True))
        for u, v, d in H.edges(data=True):
            w = d.get("weight", 1.0)

            if simple.has_edge(u, v):
                simple[u][v]["weight"] += w
            else:
                edge_attrs = dict(d)
                edge_attrs["weight"] = w
                simple.add_edge(u, v, **edge_attrs)
        return simple

    return nx.Graph(H)


def get_node_strength(G: nx.Graph, n) -> float:
    """Сумма весов всех инцидентных рёбер узла."""
    strength = 0.0
    for _, _, d in G.edges(n, data=True):
        w_raw = d.get("weight", 1.0)
        try:
            w = float(w_raw)
        except (TypeError, ValueError):
            w = 1.0
        strength += w if np.isfinite(w) else 1.0
    return float(strength)
