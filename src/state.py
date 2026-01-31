from dataclasses import dataclass, field
import time
from typing import Any

import pandas as pd


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
