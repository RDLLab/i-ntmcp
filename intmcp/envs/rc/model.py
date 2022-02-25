"""The POSG Model for the Runner Chaser Problem """
import random
from argparse import ArgumentParser
from typing import Optional, List, Tuple

import intmcp.model as M

from intmcp.envs.rc import obs as Z
from intmcp.envs.rc import state as S
from intmcp.envs.rc import action as A
from intmcp.envs.rc import grid as grid_lib


class RCModel(M.POSGModel):
    """Runner Chaser Problem Model """

    NUM_AGENTS = 2

    RUNNER_IDX = 0
    CHASER_IDX = 1

    R_ACTION = -1.0
    R_CAPTURE = -100.0    # Runner reward, Chaser = -R_CAPTURE
    R_SAFE = 100.0      # Runner reward, Chaser = -R_SAFE

    def __init__(self, grid_name: str, **kwargs):
        super().__init__(self.NUM_AGENTS, **kwargs)
        self.grid = grid_lib.load_grid(grid_name)
        self._action_space = A.get_action_space(self.NUM_AGENTS)

        self._runner_start_loc = random.choice(self.grid.init_runner_locs)
        self._chaser_start_loc = random.choice(self.grid.init_chaser_locs)

    @property
    def state_space(self) -> List[M.State]:
        raise NotImplementedError

    @property
    def obs_space(self) -> List[M.JointObservation]:
        raise NotImplementedError

    @property
    def action_space(self) -> List[M.JointAction]:
        return self._action_space

    @property
    def r_min(self) -> float:
        return self.R_CAPTURE

    @property
    def r_max(self) -> float:
        return self.R_SAFE

    @property
    def reinvigorate_fn(self) -> Optional[M.ReinvigorateFunction]:

        def rinv_fn(agent_id: M.AgentID,
                    state: M.State,
                    action: M.JointAction,  # pylint: disable=[unused-argument]
                    obs: M.Observation
                    ) -> Tuple[M.State, M.JointObservation]:
            """ReinvigorateFunction for the RC environment.

            Involves transporting the other agent to a random valid position
            on the grid, that is consistent with the observation
            """
            assert isinstance(state, S.RCState)
            assert isinstance(obs, Z.RCObs)

            if agent_id == self.RUNNER_IDX:
                # since actions are deterministic and agent knows it's own
                # start location, it's location is fully observed given history
                runner_loc = state.runner_loc

                if Z.OPPONENT in obs.adj_obs:
                    chaser_loc = self._get_opponent_loc_from_obs(obs)
                else:
                    valid_locs = self.grid.valid_chaser_locs
                    valid_locs.difference({runner_loc})
                    valid_locs.difference(
                        self.grid.get_neighbouring_locs(runner_loc, True)
                    )
                    chaser_loc = random.choice(list(valid_locs))

                runner_obs = obs
                chaser_obs = self._get_agent_obs(
                    self.CHASER_IDX, chaser_loc, runner_loc
                )
            else:
                chaser_loc = state.chaser_loc

                if Z.OPPONENT in obs.adj_obs:
                    runner_loc = self._get_opponent_loc_from_obs(obs)
                else:
                    valid_locs = self.grid.valid_runner_locs
                    valid_locs.difference_update({chaser_loc})
                    valid_locs.difference_update(
                        self.grid.get_neighbouring_locs(chaser_loc, True)
                    )

                    runner_loc = random.choice(list(valid_locs))

                chaser_obs = obs
                runner_obs = self._get_agent_obs(
                    self.RUNNER_IDX, runner_loc, chaser_loc
                )

            updated_state = S.RCState(runner_loc, chaser_loc, self.grid)
            joint_obs = M.JointObservation((runner_obs, chaser_obs))

            return updated_state, joint_obs

        return rinv_fn

    def _get_opponent_loc_from_obs(self, obs: Z.RCObs) -> grid_lib.Loc:
        # Assumes opponent is in the observation
        # An Error will be thrown if not
        opp_idx = obs.adj_obs.index(Z.OPPONENT)
        offset = [-self.grid.width, self.grid.width, 1, -1][opp_idx]
        return obs.loc + offset

    def reset(self) -> Tuple[M.State, M.JointObservation]:
        self._runner_start_loc = random.choice(self.grid.init_runner_locs)

        chaser_start_locs = list(self.grid.init_chaser_locs)
        if self._runner_start_loc in chaser_start_locs:
            chaser_start_locs.remove(self._runner_start_loc)
        self._chaser_start_loc = random.choice(chaser_start_locs)
        return super().reset()

    @property
    def initial_belief(self) -> M.Belief:
        def init_belief_fn():
            runner_loc = random.choice(self.grid.init_runner_locs)
            chaser_loc = random.choice(self.grid.init_chaser_locs)
            return S.RCState(runner_loc, chaser_loc, self.grid)
        return M.InitialParticleBelief(init_belief_fn)   # type: ignore

    def sample_initial_state(self) -> M.State:
        return self.initial_belief.sample()

    def sample_initial_obs(self, state: M.State) -> M.JointObservation:
        assert isinstance(state, S.RCState)
        return self._get_obs(state)

    def get_init_belief(self, agent_id: int, obs: M.Observation) -> M.Belief:
        assert isinstance(obs, Z.RCObs)
        if agent_id == self.RUNNER_IDX:
            return self._get_init_runner_belief(obs)
        return self._get_init_chaser_belief(obs)

    def _get_init_runner_belief(self, obs: Z.RCObs) -> M.Belief:
        def init_belief_fn():
            chaser_loc = random.choice(self.grid.init_chaser_locs)
            return S.RCState(obs.loc, chaser_loc, self.grid)
        return M.InitialParticleBelief(init_belief_fn)   # type: ignore

    def _get_init_chaser_belief(self, obs: Z.RCObs) -> M.Belief:
        def init_belief_fn():
            runner_loc = random.choice(self.grid.init_runner_locs)
            return S.RCState(runner_loc, obs.loc, self.grid)
        return M.InitialParticleBelief(init_belief_fn)   # type: ignore

    def step(self,
             state: M.State,
             action: M.JointAction
             ) -> Tuple[M.State, M.JointObservation, List[float], bool]:
        assert isinstance(state, S.RCState)
        next_state = self._get_next_state(state, action)
        obs = self._get_obs(next_state)
        reward = self._get_reward(next_state)
        done = self.is_terminal(next_state)
        return next_state, obs, reward, done

    def _get_next_state(self,
                        state: S.RCState,
                        action: M.JointAction) -> S.RCState:
        runner_loc, chaser_loc = state.runner_loc, state.chaser_loc
        runner_a = action[self.RUNNER_IDX].action_num
        chaser_a = action[self.CHASER_IDX].action_num

        chaser_next_loc = self.get_agent_next_loc(chaser_loc, chaser_a)
        if chaser_next_loc == runner_loc:
            # Runner considered to capture Fugitive
            runner_next_loc = runner_loc
        else:
            runner_next_loc = self.get_agent_next_loc(runner_loc, runner_a)

        return S.RCState(runner_next_loc, chaser_next_loc, state.grid)

    def get_agent_next_loc(self,
                           loc: grid_lib.Loc,
                           action: int) -> grid_lib.Loc:
        """Get next location given loc and action """
        coords = self.grid.loc_to_coord(loc)
        new_coords = list(coords)
        if action == grid_lib.NORTH:
            new_coords[0] = max(0, coords[0]-1)
        elif action == grid_lib.SOUTH:
            new_coords[0] = min(self.grid.height-1, coords[0]+1)
        elif action == grid_lib.EAST:
            new_coords[1] = min(self.grid.width-1, coords[1]+1)
        elif action == grid_lib.WEST:
            new_coords[1] = max(0, coords[1]-1)
        new_loc = self.grid.coord_to_loc((new_coords[0], new_coords[1]))

        if new_loc in self.grid.block_locs:
            return loc
        return new_loc

    def _get_obs(self, next_s: S.RCState) -> M.JointObservation:
        runner_loc, chaser_loc = next_s.runner_loc, next_s.chaser_loc
        runner_obs = self._get_agent_obs(
            self.RUNNER_IDX, runner_loc, chaser_loc
        )
        chaser_obs = self._get_agent_obs(
            self.CHASER_IDX, chaser_loc, runner_loc
        )
        return M.JointObservation((runner_obs, chaser_obs))

    def _get_agent_obs(self,
                       agent_id: int,
                       agent_loc: grid_lib.Loc,
                       opponent_loc: grid_lib.Loc) -> Z.RCObs:
        if agent_id == self.RUNNER_IDX:
            loc_obs = self._runner_start_loc
        else:
            loc_obs = self._chaser_start_loc
        adj_obs = self._get_adj_obs(agent_loc, opponent_loc)
        return Z.RCObs(loc_obs, adj_obs)

    def _get_adj_obs(self,
                     loc: grid_lib.Loc,
                     opponent_loc: grid_lib.Loc) -> Tuple[int, ...]:
        width, height = self.grid.width, self.grid.height
        adj_locs = [
            loc-width if loc >= width else -1,             # N
            loc+width if loc < (height-1)*width else -1,   # S
            loc+1 if loc % width < width-1 else -1,        # E
            loc-1 if loc % width > 0 else -1               # W
        ]

        adj_obs = []
        for adj_loc in adj_locs:
            if adj_loc == opponent_loc:
                adj_obs.append(Z.OPPONENT)
            elif adj_loc == -1 or adj_loc in self.grid.block_locs:
                adj_obs.append(Z.WALL)
            else:
                adj_obs.append(Z.EMPTY)

        return tuple(adj_obs)

    def _get_reward(self, state: S.RCState) -> List[float]:
        chaser_loc, runner_loc = state.chaser_loc, state.runner_loc
        if runner_loc in self.grid.safe_locs:
            return [self.R_SAFE, -self.R_SAFE]
        if (
            runner_loc == chaser_loc
            or runner_loc in self.grid.get_neighbouring_locs(chaser_loc, True)
        ):
            return [self.R_CAPTURE, -self.R_CAPTURE]
        return [self.R_ACTION, self.R_ACTION]

    def is_terminal(self, state: M.State) -> bool:
        assert isinstance(state, S.RCState)
        chaser_loc, runner_loc = state.chaser_loc, state.runner_loc
        if chaser_loc == runner_loc or runner_loc in self.grid.safe_locs:
            return True
        return runner_loc in self.grid.get_neighbouring_locs(chaser_loc, True)

    def get_outcome(self, state: M.State) -> List:
        assert isinstance(state, S.RCState)
        chaser_loc, runner_loc = state.chaser_loc, state.runner_loc
        if runner_loc in self.grid.safe_locs:
            return [M.Outcomes.WIN, M.Outcomes.LOSS]
        if (
            runner_loc == chaser_loc
            or runner_loc in self.grid.get_neighbouring_locs(chaser_loc, True)
        ):
            return [M.Outcomes.LOSS, M.Outcomes.WIN]
        return [M.Outcomes.DRAW, M.Outcomes.DRAW]

    @classmethod
    def get_args_parser(cls,
                        parser: Optional[ArgumentParser] = None
                        ) -> ArgumentParser:
        parser = super().get_args_parser(parser)
        parser.add_argument(
            "--grid_name", type=str, default='cap7',
            help="name of the grid to use env (default='cap7)"
        )
        return parser
