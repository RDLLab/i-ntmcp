"""Functions for finding shortest paths in grid """
from typing import Dict, List
from queue import PriorityQueue

import intmcp.envs.pe.grid as grid_lib

Loc = grid_lib.Loc
Grid = grid_lib.Grid


def all_shortest_paths(src_locs: List[Loc],
                       grid: Grid) -> Dict[Loc, Dict[Loc, float]]:
    """Get shortest paths from each src loc to each other loc in grid """
    src_dists = {}
    for src_loc in src_locs:
        src_dists[src_loc] = dijkstra(src_loc, grid)
    return src_dists


def dijkstra(src_loc: Loc, grid: Grid) -> Dict[Loc, float]:
    """Get shortest paths between source loc and all other locs in grid """
    dist = {src_loc: 0.0}
    pq = PriorityQueue()
    pq.put((dist[src_loc], src_loc))

    visited = set([src_loc])

    while not pq.empty():
        d, loc = pq.get()
        for adj_loc in grid.get_neighbouring_locs(loc, False):
            if dist[loc] + 1 < dist.get(adj_loc, float('inf')):
                dist[adj_loc] = dist[loc] + 1
                if adj_loc not in visited:
                    pq.put((dist[adj_loc], adj_loc))
                    visited.add(adj_loc)
    return dist
