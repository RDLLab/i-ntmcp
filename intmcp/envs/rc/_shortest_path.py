"""Utility module for finding shortest path between locs in RC env """
import sys
import queue
import collections
from typing import Dict, Tuple

import intmcp.model as M

from intmcp.envs.rc import action as A
from intmcp.envs.rc import grid as grid_lib
import intmcp.envs.rc.model as rc_model_lib


def extract_policy(start_loc: grid_lib.Loc,
                   target_loc: grid_lib.Loc,
                   came_from: Dict[grid_lib.Loc, Tuple[grid_lib.Loc, int]]
                   ) -> Dict[grid_lib.Loc, A.RCAction]:
    """Extract policy from A* path map """
    current_loc = target_loc
    policy = {target_loc: A.RCAction(0)}
    while current_loc != start_loc:
        current_loc, action_num = came_from[current_loc]
        policy[current_loc] = A.RCAction(action_num)
    return policy


def a_star_search(start_loc: grid_lib.Loc,
                  target_loc: grid_lib.Loc,
                  agent_id: M.AgentID,
                  model: rc_model_lib.RCModel):
    """Run A* Search """
    # pylint: disable=[protected-access]
    grid = model._grid

    action_space = []
    for a in model.get_agent_action_space(agent_id):
        assert isinstance(a, A.RCAction)
        action_space.append(a.action_num)

    frontier = queue.PriorityQueue()   # type: ignore
    frontier.put((0, start_loc))

    came_from = {start_loc: (start_loc, -1)}

    g_score = collections.defaultdict(lambda: sys.maxsize)
    g_score[start_loc] = 0

    while not frontier.empty():

        current_loc = frontier.get()[1]
        if current_loc == target_loc:
            return extract_policy(start_loc, target_loc, came_from)

        for action in action_space:
            next_loc = model._get_agent_next_loc(current_loc, action)

            new_g_score = g_score[current_loc] + 1
            if new_g_score < g_score[next_loc]:
                came_from[next_loc] = (current_loc, action)
                g_score[next_loc] = new_g_score
                priority = (
                    new_g_score + grid.manhattan_dist(next_loc, target_loc)
                )
                frontier.put((priority, next_loc))

    raise ValueError(f"Not path found from {start_loc=} to {target_loc=}")
