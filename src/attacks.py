import math
import random
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx

from .metrics import calculate_metrics, add_dist_attr
from .utils import as_simple_undirected, get_node_strength


def _pick_nodes_adaptive(
    H: nx.Graph,
    attack_kind: str,
    k: int,
    rng: np.random.Generator,
) -> Optional[list]:
    """
    Pick k nodes from the CURRENT graph H according to adaptive strategy.
    Returns None for unsupported strategies to fall back to existing code.
    """
    if k <= 0 or H.number_of_nodes() == 0:
        return []

    nodes = list(H.nodes())

    if attack_kind == "random":
        rng.shuffle(nodes)
        return nodes[:k]

    if attack_kind == "low_degree":
        nodes.sort(key=lambda n: H.degree(n))
        return nodes[:k]

    if attack_kind == "weak_strength":
        nodes.sort(key=lambda n: get_node_strength(H, n))
        return nodes[:k]

    return None

# =========================
# Rich-Club helpers
# =========================
def strength_ranking(G: nx.Graph) -> list:
    """Rank nodes by weighted degree (strength)."""
    strength = dict(G.degree(weight="weight"))
    nodes = list(G.nodes())
    nodes_sorted = sorted(nodes, key=lambda n: strength.get(n, 0.0), reverse=True)
    return nodes_sorted


def richclub_top_fraction(G: nx.Graph, rc_frac: float) -> list:
    """Return top fraction of nodes by strength."""
    nodes_sorted = strength_ranking(G)
    if not nodes_sorted:
        return []
    k = max(1, int(len(nodes_sorted) * float(rc_frac)))
    return nodes_sorted[:k]


def richclub_by_density_threshold(G: nx.Graph, min_density: float, max_frac: float) -> list:
    """Return the largest prefix with induced density above threshold."""
    nodes_sorted = strength_ranking(G)
    n = len(nodes_sorted)
    if n == 0:
        return []
    if n < 3:
        return nodes_sorted

    maxK = max(3, int(n * float(max_frac)))
    maxK = min(maxK, n)

    best = nodes_sorted[:3]
    for K in range(3, maxK + 1):
        club = nodes_sorted[:K]
        H = G.subgraph(club)
        dens = nx.density(H)
        if dens >= float(min_density):
            best = club
    return best


def pick_targets_for_attack(
    G: nx.Graph,
    attack_kind: str,
    step_size: int,
    seed: int,
    rc_frac: float,
    rc_min_density: float,
    rc_max_frac: float,
) -> list:
    """Select nodes to remove per attack strategy."""
    nodes = list(G.nodes())
    if not nodes:
        return []

    rng = random.Random(int(seed))

    if attack_kind == "random":
        k = min(len(nodes), step_size)
        return rng.sample(nodes, k)

    if attack_kind == "degree":
        strength = dict(G.degree(weight="weight"))
        return sorted(nodes, key=lambda n: strength.get(n, 0.0), reverse=True)[:step_size]

    if attack_kind == "betweenness":
        H = add_dist_attr(G)
        n = H.number_of_nodes()
        # Aggressive sampling: k ~= sqrt(n) capped for speed on large graphs.
        k_samples = min(int(math.sqrt(n)) + 1, 100, n)
        bc = nx.betweenness_centrality(H, k=k_samples, weight="dist", normalized=True, seed=int(seed))
        return sorted(nodes, key=lambda n: bc.get(n, 0.0), reverse=True)[:step_size]

    if attack_kind == "kcore":
        try:
            core = nx.core_number(G)
        except Exception:
            core = {n: 0 for n in nodes}
        return sorted(nodes, key=lambda n: core.get(n, 0), reverse=True)[:step_size]

    if attack_kind == "richclub_top":
        club = richclub_top_fraction(G, rc_frac=rc_frac)
        if not club:
            return []
        return club[:min(step_size, len(club))]

    if attack_kind == "richclub_density":
        club = richclub_by_density_threshold(G, min_density=rc_min_density, max_frac=rc_max_frac)
        if not club:
            return []
        return club[:min(step_size, len(club))]

    return []


def lcc_fraction(G: nx.Graph, N0: int) -> float:
    """Compute fraction of nodes in the largest connected component."""
    if G.number_of_nodes() == 0 or N0 <= 0:
        return 0.0
    lcc = len(max(nx.connected_components(G), key=len))
    return float(lcc) / float(N0)

# =========================
# Attack simulation
# =========================
def run_attack(
    G_in: nx.Graph,
    attack_kind: str,
    remove_frac: float,
    steps: int,
    seed: int,
    eff_sources_k: int,
    rc_frac: float = 0.10,
    rc_min_density: float = 0.30,
    rc_max_frac: float = 0.30,
    compute_heavy_every: int = 1,
    keep_states: bool = False,
):
    """Unified runner for both static (centrality) and adaptive (weak nodes) attacks."""
    G = as_simple_undirected(G_in).copy()
    N0 = G.number_of_nodes()

    if N0 < 2:
        return pd.DataFrame(), {"removed_nodes": [], "states": []}

    total_to_remove = int(N0 * float(remove_frac))
    is_adaptive = attack_kind in ("low_degree", "weak_strength", "random")

    # Pre-calculate targets for static strategies.
    static_targets = []
    if not is_adaptive:
        static_targets = pick_targets_for_attack(
            G, attack_kind, total_to_remove, int(seed), rc_frac, rc_min_density, rc_max_frac
        )

    # Simulation loop.
    ks = np.linspace(0, total_to_remove, int(steps) + 1).round().astype(int).tolist()
    np_rng = np.random.default_rng(int(seed))
    history, states, removed_log = [], [], []

    for i, target_k in enumerate(ks):
        if G.number_of_nodes() == 0 and i > 0:
            break

        if keep_states:
            states.append(G.copy())

        # Metrics snapshot.
        heavy = (i % max(1, int(compute_heavy_every)) == 0)
        met = (
            calculate_metrics(G, int(eff_sources_k), int(seed), False)
            if heavy
            else {"N": G.number_of_nodes(), "E": G.number_of_edges()}
        )
        met.update(
            {
                "step": i,
                "removed_frac": len(removed_log) / N0,
                "lcc_frac": lcc_fraction(G, N0),
            }
        )
        history.append(met)

        # Node removal.
        if i < len(ks) - 1:
            num_to_del = ks[i + 1] - len(removed_log)
            if num_to_del > 0:
                if is_adaptive:
                    targets = _pick_nodes_adaptive(G, attack_kind, num_to_del, np_rng) or []
                else:
                    targets = static_targets[len(removed_log) : ks[i + 1]]

                G.remove_nodes_from(targets)
                removed_log.extend(targets)

    return pd.DataFrame(history), {"removed_nodes": removed_log, "states": states}
