"""Value functions for the Pursuit evasion environment """
import random
from typing import Dict, Tuple, Optional, List

import intmcp.model as M
import intmcp.policy as policy_lib

import intmcp.envs.pe.obs as Z
import intmcp.envs.pe.action as A
import intmcp.envs.pe.grid as grid_lib
from intmcp.envs.pe.model import PEModel
import intmcp.envs.pe.shortest_path as sp_lib


Loc = grid_lib.Loc


class PESPPolicy(policy_lib.BasePolicy):
    """Preferred Action policy for Runner in the PE environment

    This policy sets the preferred action as the one which is on the shortest
    path to the runners goal and which doesnt leave agent in same position.
    """

    SP_N_INIT = 10

    def __init__(self,
                 model: PEModel,
                 ego_agent: M.AgentID,
                 gamma: float,
                 r_hi: Optional[float] = None,
                 r_lo: Optional[float] = None,
                 **kwargs):
        super().__init__(model, ego_agent, gamma, **kwargs)
        assert isinstance(self.model, PEModel)
        assert (
            (r_hi is None and r_lo is None)
            or (r_hi is not None and r_lo is not None and r_hi >= r_lo)
        )

        self._grid = self.model.grid
        self._r_hi = r_hi
        self._r_lo = r_lo
        self._is_runner = self.ego_agent == self.model.RUNNER_IDX

        runner_end_locs = list(set(
            self._grid.runner_start_locs + self._grid.runner_goal_locs
        ))
        self._dists = sp_lib.all_shortest_paths(runner_end_locs, self._grid)

        self._loc = runner_end_locs[0]
        self._prev_loc = -1
        self._update_num = 0

    def get_action(self) -> M.Action:
        _, obs = self.history.get_last_step()
        if self._is_runner:
            assert isinstance(obs, Z.PERunnerObs)
            return self._sample_sp_action(self._loc, self._prev_loc, obs.goal)
        assert isinstance(obs, Z.PEChaserObs)
        return self._sample_sp_action(
            self._loc,
            self._prev_loc,
            self.model.runner_start_loc   # type: ignore
        )

    def get_action_by_history(self, history: M.AgentHistory) -> M.Action:
        _, obs = history.get_last_step()
        loc, prev_loc = self._get_loc_from_history(history)

        if self._is_runner:
            assert isinstance(obs, Z.PERunnerObs)
            goal_loc = obs.goal
        else:
            assert isinstance(self.model, PEModel)
            goal_loc = self.model.runner_start_loc

        return self._get_sp_action(loc, prev_loc, goal_loc)

    def get_pi_by_history(self,
                          history: Optional[M.AgentHistory] = None
                          ) -> policy_lib.ActionDist:
        loc, prev_loc, goal_loc = self._get_loc_and_goal_loc(history)
        dists = self._get_action_sp_dists(loc, prev_loc, goal_loc)

        pi = {a: 1.0 for a in self.action_space}
        max_dist, min_dist = max(dists), min(dists)
        dist_gap = max(1, max_dist - min_dist)

        weight_sum = 0.0
        for i, a in enumerate(self.action_space):
            weight = 1 - ((dists[i] - min_dist) / dist_gap)
            pi[a] = weight
            weight_sum += weight

        for a in self.action_space:
            pi[a] /= weight_sum

        return pi

    def get_action_init_values(self,
                               history: M.AgentHistory
                               ) -> Dict[M.Action, Tuple[float, int]]:
        assert isinstance(self.model, PEModel)
        if self._is_runner:
            terminal_reward = self.model.R_EVASION
        else:
            terminal_reward = self.model.R_CAPTURE

        loc, prev_loc, goal_loc = self._get_loc_and_goal_loc(history)
        dists = self._get_action_sp_dists(loc, prev_loc, goal_loc)

        # Bias actions towards shortest path
        # Assign r_hi to actions in direction of shortest path
        # Assign r_lo to actions in direction of longest path
        # Assign weighted value between r_hi and r_lo for other actions
        # Similarly for n_init with self.SP_N_INIT
        init_values = {}
        max_dist, min_dist = max(dists), min(dists)
        dist_gap = max(1, max_dist - min_dist)

        if self._r_hi is not None:
            r_hi = self._r_hi
        else:
            r_hi = terminal_reward - min_dist

        if self._r_lo is None:
            r_lo = min(r_hi, -terminal_reward)
        else:
            r_lo = self._r_lo

        for i, a in enumerate(self.action_space):
            weight = 1 - ((dists[i] - min_dist) / dist_gap)
            v_init = r_hi - (1.0-weight)*(r_hi - r_lo)
            n_init = weight * self.SP_N_INIT
            init_values[a] = (v_init, int(n_init))

        return init_values

    def get_value(self, history: Optional[M.AgentHistory]) -> float:
        return 0.0

    def update(self, action: M.Action, obs: M.Observation) -> None:
        super().update(action, obs)
        assert isinstance(self.model, PEModel)

        if self._update_num == 0:
            if self._is_runner:
                self._loc = self.model.runner_start_loc
            else:
                self._loc = self.model.chaser_start_loc
        else:
            assert isinstance(action, A.PEAction)
            a_num = action.action_num
            self._prev_loc = self._loc
            self._loc = self.model.get_agent_next_loc(a_num, self._loc)

        self._update_num += 1

    def reset(self) -> None:
        super().reset()
        assert isinstance(self.model, PEModel)
        self._update_num = 0

        if self._is_runner:
            self._loc = self.model.runner_start_loc
        else:
            self._loc = self.model.chaser_start_loc

    def _get_sp_action(self,
                       loc: Loc,
                       prev_loc: Loc,
                       goal_loc: Loc) -> A.PEAction:
        sp_action = A.NORTH
        sp_dist = float('inf')
        for a in A.ACTIONS:
            new_loc = self._grid.get_neighbour(loc, a, False)
            if new_loc is None or new_loc == prev_loc:
                continue
            dist = self._get_sp_dist(new_loc, goal_loc)
            if dist < sp_dist:
                sp_dist = dist
                sp_action = a
        return A.PEAction(sp_action)

    def _sample_sp_action(self,
                          loc: Loc,
                          prev_loc: Loc,
                          goal_loc: Loc) -> A.PEAction:
        dists = self._get_action_sp_dists(loc, prev_loc, goal_loc)

        pi: List[float] = []
        max_dist, min_dist = max(dists), min(dists)
        dist_gap = max(1, max_dist - min_dist)

        weight_sum = 0.0
        # Use cumulative prob dist since it's faster to sample
        # and ~same cost to create
        for i in range(len(self.action_space)):
            weight = 1 - ((dists[i] - min_dist) / dist_gap)
            pi.append(
                weight if i == 0 else pi[-1]+weight
            )
            weight_sum += weight

        for i in range(len(self.action_space)):
            pi[i] /= weight_sum

        return random.choices(   # type: ignore
            self.action_space, cum_weights=pi, k=1
        )[0]

    def _get_action_sp_dists(self,
                             loc: Loc,
                             prev_loc: Loc,
                             goal_loc: Loc) -> List[float]:
        dists = []
        for a in self.action_space:
            assert isinstance(a, A.PEAction)
            new_loc = self._grid.get_neighbour(loc, a.action_num, False)
            if new_loc is None or new_loc == prev_loc:
                # action leaves position unchanged or reverses direction
                # so penalize distance by setting it to max possible dist
                d = self._get_sp_dist(loc, goal_loc)
                d += 2
            else:
                d = self._get_sp_dist(new_loc, goal_loc)
            dists.append(d)
        return dists

    def _get_sp_dist(self, loc: Loc, goal_loc: Loc) -> float:
        return self._dists[goal_loc][loc]

    def _get_loc_and_goal_loc(self,
                              history: Optional[M.AgentHistory]
                              ) -> Tuple[Loc, Loc, Loc]:
        assert isinstance(self.model, PEModel)

        if history is None:
            _, obs = self.history.get_last_step()
            loc = self._loc
            prev_loc = self._prev_loc
        else:
            _, obs = history.get_last_step()
            loc, prev_loc = self._get_loc_from_history(history)

        if self._is_runner:
            assert isinstance(obs, Z.PERunnerObs)
            goal_loc = obs.goal
        else:
            assert isinstance(obs, Z.PEChaserObs)
            goal_loc = self.model.runner_start_loc

        return loc, prev_loc, goal_loc

    def _get_loc_from_history(self,
                              history: M.AgentHistory) -> Tuple[Loc, Loc]:
        assert isinstance(self.model, PEModel)

        if self._is_runner:
            loc = self.model.runner_start_loc
        else:
            loc = self.model.chaser_start_loc

        prev_loc = -1
        for (a, _) in history:
            if isinstance(a, M.NullAction):
                continue
            a_num = a.action_num
            prev_loc = loc
            loc = self.model.get_agent_next_loc(a_num, loc)
        return loc, prev_loc
