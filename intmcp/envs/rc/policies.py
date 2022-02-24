"""Policies for the Runner Chaser environment """
import math
from argparse import ArgumentParser
from typing import Union, List, Optional, Dict, Tuple

import intmcp.model as M
import intmcp.policy as policy_lib

import intmcp.envs.rc.obs as Z
import intmcp.envs.rc.action as A
import intmcp.envs.rc.grid as grid_lib
from intmcp.envs.rc.model import RCModel
import intmcp.envs.rc._shortest_path as pathfinding_lib


class RCNestedReasoningPolicy(policy_lib.BasePolicy):
    """A nested reasoning policy for the Runner Chaser environment

    NOTE: This policy is specific to the cap7 grid.

    Specifically selects actions for a given observation based on a fixed
    nested reasoning level l, as follows:

    Let:

    - G0 = closest runner goal to runner start
    - G1 = 2nd closest runner goal to runner start
    - P0 = path from chaser start to runner start via G0
    - P1 = path from chaser start to runner start via G1

    N.B. The shortest path from chaser start to runner start is P0

    RUNNER
    if math.ceil(l / 2) % 2 == 0:      # 0, 3, 4, 7, 8
       takes actions towards G0
    if math.ceil(l / 2) % 2 == 1:      # 1, 2, 5, 6, 9, 10, ..
       takes actions towards G1

    CHASER
    if l // 2 % 2 == 0:                # 0, 1, 4, 5, 8, 9
       takes actions along P0
    if l // 2 % 2 == 1:                # 2, 3, 6, 7, 10, 11, ...
       takes actions along P1

    If the chaser does not find the runner along the chosen path and the
    episode has not ended then it will move along the other path (except in
    the opposite direction)

    In table form:

     l | R   | C  |
    ---------------
     0 | G0  | G0 |
     1 | G1  | G0 |
     2 | G1  | G1 |
     3 | G0  | G1 |
     4 | G0  | G0 |
     5 | G1  | G0 |
    """

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: M.AgentID,
                 gamma: float,
                 nesting_level: int,
                 **kwargs):
        super().__init__(model, ego_agent, gamma, **kwargs)
        self.nesting_level = nesting_level

        assert isinstance(self.model, RCModel)
        self._grid = self.model.grid
        self._is_runner = self.ego_agent == self.model.RUNNER_IDX

        runner_start_loc = self._grid.init_runner_locs[0]
        self._closest_safe_locs = self._get_closest_safe_locs(runner_start_loc)

        self._loc = runner_start_loc
        self._update_num = 0

        self._loc_order = self._get_locs_order(nesting_level)
        self._policy = self._compute_policy(self._loc_order)

    def get_action(self) -> M.Action:
        return self._policy[self._loc]

    def get_action_by_history(self, history: M.AgentHistory) -> M.Action:
        loc = self._get_loc_from_history(history)
        return self._policy[loc]

    def get_action_init_values(self,
                               history: M.AgentHistory
                               ) -> Dict[M.Action, Tuple[float, int]]:
        return {a: (0.0, 0) for a in self.action_space}

    def get_value(self, history: Optional[M.AgentHistory]) -> float:
        return 0.0

    def update(self, action: M.Action, obs: M.Observation) -> None:
        super().update(action, obs)
        assert isinstance(obs, Z.RCObs)

        if self._update_num == 0:
            # pylint: disable=[pointless-statement]
            # statement is not pointless...
            self._loc == obs.loc
        else:
            assert isinstance(action, A.RCAction)
            self._loc = self.model.get_agent_next_loc(   # type: ignore
                self._loc, action.action_num
            )

        if self._loc == self._loc_order[-1]:
            # reached end of the path so traverse path in opposite dir
            self._loc_order.reverse()
            self._policy = self._compute_policy(self._loc_order)

        self._update_num += 1

    def reset(self) -> None:
        super().reset()
        self._update_num = 0

        if self._is_runner:
            self._loc = self._grid.init_runner_locs[0]
        else:
            self._loc = self._grid.init_chaser_locs[0]

        self._loc_order = self._get_locs_order(self.nesting_level)
        self._policy = self._compute_policy(self._loc_order)

    def _get_loc_from_history(self, history: M.AgentHistory) -> grid_lib.Loc:
        assert isinstance(self.model, RCModel)

        if self._is_runner:
            loc = self._grid.init_runner_locs[0]
        else:
            loc = self._grid.init_chaser_locs[0]

        for (a, _) in history:
            if isinstance(a, M.NullAction):
                continue
            a_num = a.action_num
            loc = self.model.get_agent_next_loc(a_num, loc)
        return loc

    def _get_locs_order(self,
                        nesting_level: int
                        ) -> List[grid_lib.Loc]:
        runner_start_loc = self._grid.init_runner_locs[0]
        chaser_start_loc = self._grid.init_chaser_locs[0]

        if self._is_runner:
            if math.ceil(nesting_level / 2) % 2 == 0:
                target_loc = self._closest_safe_locs[0]
            else:
                target_loc = self._closest_safe_locs[1]
            return [runner_start_loc, target_loc]

        if nesting_level // 2 % 2 == 0:
            return [
                chaser_start_loc,
                self._closest_safe_locs[0],
                runner_start_loc,
                self._closest_safe_locs[1]
            ]
        return [
            chaser_start_loc,
            self._closest_safe_locs[1],
            runner_start_loc,
            self._closest_safe_locs[0]
        ]

    def _get_closest_safe_locs(self,
                               origin_loc: grid_lib.Loc,
                               ) -> Tuple[grid_lib.Loc, grid_lib.Loc]:
        safe_locs = list(self._grid.safe_locs)
        safe_dists = [
            self._grid.manhattan_dist(origin_loc, loc) for loc in safe_locs
        ]
        sorted_dists = sorted(safe_dists)
        return (
            safe_locs[safe_dists.index(sorted_dists[0])],
            safe_locs[safe_dists.index(sorted_dists[1])]
        )

    def _compute_policy(self,
                        loc_order: List[grid_lib.Loc]
                        ) -> Dict[grid_lib.Loc, A.RCAction]:
        assert isinstance(self.model, RCModel)
        policy = {}
        for i in range(len(loc_order)-1):
            intermediate_policy = pathfinding_lib.a_star_search(
                loc_order[i], loc_order[i+1], self.ego_agent, self.model
            )
            policy.update(intermediate_policy)
        return policy

    @classmethod
    def get_args_parser(cls,
                        parser: Optional[ArgumentParser] = None
                        ) -> ArgumentParser:
        parser = super().get_args_parser(parser)
        parser.add_argument(
            "--nesting_level", nargs="*", type=int, default=[1],
            help="Number of nesting levels (default=[1])"
        )
        return parser

    # pylint: disable=[arguments-differ]
    @classmethod
    def initialize(cls,
                   model: M.POSGModel,
                   ego_agent: int,
                   gamma: float,
                   nesting_level: Union[int, List[int]] = 1,
                   **kwargs) -> 'RCNestedReasoningPolicy':
        if isinstance(nesting_level, list):
            if len(nesting_level) == 1:
                nesting_level = nesting_level[0]
            else:
                nesting_level = nesting_level[ego_agent]

        return cls(model, ego_agent, gamma, nesting_level, **kwargs)
