import streamlit as st
import networkx as nx
import pandas as pd

from src.core_math import fragility_from_curvature, ollivier_ricci_summary
from src.graph_wrapper import GraphWrapper


@st.cache_resource(show_spinner=False)
def build_graph(df: pd.DataFrame) -> nx.Graph:
    """Build a NetworkX graph from a pandas edge list (cached by dataframe content)."""
    return nx.from_pandas_edgelist(df, "src", "dst", edge_attr=True)


@st.cache_data(show_spinner=False)
def compute_layout(wrapper: GraphWrapper) -> dict:
    """Compute 2D layout. Instant cache check via GraphWrapper hash."""
    return nx.spring_layout(wrapper.G, seed=42)


@st.cache_data(show_spinner=False)
def compute_curvature(
    wrapper: GraphWrapper,
    sample_edges: int = 150,
    seed: int = 42,
    max_support: int = 60,
    cutoff: float = 8.0,
) -> dict:
    """Compute Ricci curvature summary using GraphWrapper for O(1) caching."""
    G = wrapper.G
    if G.number_of_edges() == 0:
        return {
            "kappa_mean": float("nan"),
            "kappa_median": float("nan"),
            "kappa_frac_negative": float("nan"),
            "kappa_computed_edges": 0,
            "kappa_skipped_edges": 0,
            "fragility_kappa": float("nan"),
        }

    try:
        curv = ollivier_ricci_summary(
            G,
            sample_edges=int(sample_edges),
            seed=int(seed),
            max_support=int(max_support),
            cutoff=float(cutoff),
        )
        kappa_mean = float(curv.kappa_mean)
        return {
            "kappa_mean": kappa_mean,
            "kappa_median": float(curv.kappa_median),
            "kappa_frac_negative": float(curv.kappa_frac_negative),
            "kappa_computed_edges": int(curv.computed_edges),
            "kappa_skipped_edges": int(curv.skipped_edges),
            "fragility_kappa": float(fragility_from_curvature(kappa_mean)),
        }
    except Exception:
        return {"kappa_mean": float("nan"), "kappa_computed_edges": 0}
