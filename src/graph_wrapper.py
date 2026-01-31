import time
import uuid

import networkx as nx


class GraphWrapper:
    """Small wrapper that provides a fast, stable hash for Streamlit caching."""

    def __init__(self, G: nx.Graph, name: str, source: str):
        self._G = G
        self.name = name
        self.source = source
        self.id = uuid.uuid4().hex
        self.last_modified = time.time()

    @property
    def G(self) -> nx.Graph:
        """Expose the underlying NetworkX graph."""
        return self._G

    def update_graph(self, new_G: nx.Graph) -> None:
        """Explicitly update the graph and bump the cache version."""
        self._G = new_G
        self.last_modified = time.time()
        self.id = uuid.uuid4().hex

    def __hash__(self) -> int:
        """Hash only a lightweight version marker to keep caching O(1)."""
        return hash((self.id, self.last_modified))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphWrapper):
            return False
        return self.id == other.id
