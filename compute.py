import streamlit as st

import networkx as nx
import numpy as np
import pandas as pd

from src.core_math import fragility_from_curvature, ollivier_ricci_summary
from src.config import settings
from src.graph_wrapper import GraphWrapper


@st.cache_resource(show_spinner=False)
def build_graph(df: pd.DataFrame) -> nx.Graph:
    """Build a NetworkX graph from a pandas edge list.

    This is intentionally cached because turning large dataframes into graphs
    is one of the most expensive UI steps.
    """
    return nx.from_pandas_edgelist(df, "src", "dst", edge_attr=True)


@st.cache_data(show_spinner=False)
def compute_layout(wrapper: GraphWrapper) -> dict:
    """Compute and cache a deterministic 2D layout for quick preview plots."""
    return nx.spring_layout(wrapper.G, seed=settings.DEFAULT_SEED)


@st.cache_data(show_spinner=False)
def compute_curvature(
    G: nx.Graph,
    sample_edges: int = 150,
    seed: int = settings.DEFAULT_SEED,
    max_support: int = settings.RICCI_MAX_SUPPORT,
    cutoff: float = settings.RICCI_CUTOFF,
) -> dict:
    """Compute Ollivier–Ricci curvature summary metrics for a graph.

    The return shape mirrors the metrics dict used in the app so results can be
    merged into the cached metrics payload when the user explicitly requests it.
    """
    if G.number_of_edges() == 0:
        return {
            "kappa_mean": float("nan"),
            "kappa_median": float("nan"),
            "kappa_frac_negative": float("nan"),
            "kappa_computed_edges": 0,
            "kappa_skipped_edges": 0,
            "fragility_kappa": float("nan"),
        }

    curv = ollivier_ricci_summary(
        G,
        sample_edges=sample_edges,
        seed=seed,
        max_support=max_support,
        cutoff=cutoff,
    )

    kappa_mean = curv.kappa_mean
    fragility_kappa = fragility_from_curvature(kappa_mean)

    return {
        "kappa_mean": kappa_mean,
        "kappa_median": curv.kappa_median,
        "kappa_frac_negative": curv.kappa_frac_negative,
        "kappa_computed_edges": curv.computed_edges,
        "kappa_skipped_edges": curv.skipped_edges,
        "fragility_kappa": fragility_kappa,
    }
