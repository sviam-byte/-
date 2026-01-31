from dataclasses import dataclass, field
import time
from typing import Any

import pandas as pd

from .preprocess import filter_edges

@dataclass
class ExperimentData:
    """Serializable experiment payload stored in session state."""

    id: str
    name: str
    graph_id: str
    attack_kind: str
    params: dict[str, Any] = field(default_factory=dict)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    created_at: float = field(default_factory=time.time)


@dataclass
class GraphEntry:
    """Graph metadata container stored in session state."""

    id: str
    name: str
    source: str
    edges_df: pd.DataFrame
    meta_tags: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def get_filtered_df(self, min_conf: float, min_weight: float) -> pd.DataFrame:
        """Encapsulate filtering logic."""
        src = self.meta_tags.get("src_col", "src")
        dst = self.meta_tags.get("dst_col", "dst")
        return filter_edges(self.edges_df, src, dst, min_conf, min_weight)
