from typing import Dict, Any

from .node import Node
from .nst import NestedSearchTree, HistoryDist, NestingLevel, NestedBelief


SEARCH_TREES: Dict[str, Any] = {
    'NST': NestedSearchTree,
    **{
        c.__name__: c for c in [
            NestedSearchTree,
        ]
    }
}
