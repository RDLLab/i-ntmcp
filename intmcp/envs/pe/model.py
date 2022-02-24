"""The POSG Model for the Discrete Pursuit Evasion Problem """
import random
from argparse import ArgumentParser
from typing import Optional, List, Tuple

import intmcp.model as M

from intmcp.envs.pe import obs as Z
from intmcp.envs.pe import state as S
from intmcp.envs.pe import action as A
from intmcp.envs.pe import grid as grid_lib


class PEModel(M.POSGModel):
    """Discrete Pursuit Evasion Model """

    NUM_AGENTS = 2

    RUNNER_IDX = 0
    CHASER_IDX = 1

    R_RUNNER_ACTION = -1.0        # Reward each step for runner
    R_CHASER_ACTION = -1.0       # Reward each step for chaser
    R_CAPTURE = 100.0            # Chaser reward for capturing runner
    R_EVASION = 100.0            # Runner reward for reaching goal

    FOV_EXPANSION_INCR = 3
    HEARING_DIST = 2

    def __init__(self, grid_name: str, **kwargs):
        super().__init__(self.NUM_AGENTS, **kwargs)
        self._grid_name = grid_name
        self.grid = grid_lib.load_grid(grid_name)
        self._action_space = A.get_action_space(self.NUM_AGENTS)

        self._runner_start_loc = random.choice(self.grid.runner_start_locs)
        self._chaser_start_loc = random.choice(self.grid.chaser_start_locs)

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
        return -self.R_CAPTURE

    @property
    def r_max(self) -> float:
        return self.R_CAPTURE

    @property
    def reinvigorate_fn(self) -> Optional[M.ReinvigorateFunction]:

        def rinv_fn(agent_id: M.AgentID,
                    state: M.State,
                    action: M.JointAction,  # pylint: disable=[unused-argument]
                    obs: M.Observation
                    ) -> Tuple[M.State, M.JointObservation]:
            """Reinvigorate Function for the PE environment.

            Involves transporting the other agent to a random valid position
            on the grid that is consistent with the observation
            """
            assert isinstance(state, S.PEState)

            if agent_id == self.RUNNER_IDX:
                assert isinstance(obs, Z.PERunnerObs)
                # since actions are deterministic and agent knows it's own
                # start location, it's location and dir are fully observed
                # given history
                runner_loc, runner_dir = state.runner_loc, state.runner_dir
                cur_chaser_loc = state.chaser_loc
                chaser_loc = self._get_opponent_loc_from_obs(
                    runner_loc, runner_dir, cur_chaser_loc, obs.seen, obs.heard
                )
                chaser_dir = random.choice(grid_lib.DIRS)
            else:
                assert isinstance(obs, Z.PEChaserObs)
                chaser_loc, chaser_dir = state.chaser_loc, state.chaser_dir
                cur_runner_loc = state.runner_loc
                runner_loc = self._get_opponent_loc_from_obs(
                    chaser_loc, chaser_dir, cur_runner_loc, obs.seen, obs.heard
                )
                runner_dir = random.choice(grid_lib.DIRS)

            updated_state = S.PEState(
                runner_loc,
                chaser_loc,
                runner_dir,
                chaser_dir,
                state.runner_goal_loc,
                self.grid
            )

            if agent_id == self.RUNNER_IDX:
                runner_obs = obs
                chaser_obs = self._get_chaser_obs(updated_state)
            else:
                runner_obs = self._get_runner_obs(updated_state)
                chaser_obs = obs   # type: ignore

            joint_obs = M.JointObservation((runner_obs, chaser_obs))

            return updated_state, joint_obs

        return rinv_fn

    def _get_opponent_loc_from_obs(self,
                                   ego_loc: grid_lib.Loc,
                                   ego_dir: grid_lib.Loc,
                                   opp_current_loc: grid_lib.Loc,
                                   opp_seen: bool,
                                   opp_heard: bool) -> grid_lib.Loc:
        fov = grid_lib.get_fov(ego_loc, ego_dir, self.grid)
        if opp_seen:
            valid_locs = fov
        else:
            valid_locs = self.grid.valid_locs.difference(fov)

        if opp_heard:
            heard_locs = self.grid.get_locs_within_dist(
                ego_loc, self.HEARING_DIST, False, True
            )
            valid_locs = valid_locs.intersection(heard_locs)

        if opp_current_loc in valid_locs:
            return opp_current_loc

        if len(valid_locs) == 0:
            print(f"{ego_loc=} {ego_dir=} {opp_current_loc=} {opp_seen=} {opp_heard=}")
            raise AssertionError()

        closest_locs = self.grid.get_min_dist_locs(opp_current_loc, valid_locs)
        return random.choice(closest_locs)

    def reset(self) -> Tuple[M.State, M.JointObservation]:
        self._runner_start_loc = random.choice(self.grid.runner_start_locs)

        chaser_start_locs = list(self.grid.chaser_start_locs)
        if self._runner_start_loc in chaser_start_locs:
            chaser_start_locs.remove(self._runner_start_loc)
        self._chaser_start_loc = random.choice(chaser_start_locs)
        return super().reset()

    @property
    def initial_belief(self) -> M.Belief:
        def init_belief_fn():
            runner_goal_locs = self.grid.get_runner_goal_locs(
                self._runner_start_loc
            )
            runner_goal_locs = list(runner_goal_locs)
            for loc in [self._runner_start_loc, self._chaser_start_loc]:
                if loc in runner_goal_locs:
                    runner_goal_locs.remove(loc)
            runner_goal_loc = random.choice(runner_goal_locs)

            return S.PEState(
                runner_loc=self._runner_start_loc,
                chaser_loc=self._chaser_start_loc,
                runner_dir=grid_lib.NORTH,
                chaser_dir=grid_lib.NORTH,
                runner_goal_loc=runner_goal_loc,
                grid=self.grid
            )
        return M.InitialParticleBelief(init_belief_fn)   # type: ignore

    def sample_initial_state(self) -> M.State:
        return self.initial_belief.sample()

    def sample_initial_obs(self, state: M.State) -> M.JointObservation:
        assert isinstance(state, S.PEState)
        obs, _ = self._get_obs(state)
        return obs

    def get_init_belief(self, agent_id: int, obs: M.Observation) -> M.Belief:
        if agent_id == self.RUNNER_IDX:
            assert isinstance(obs, Z.PERunnerObs)
            return self._get_init_runner_belief(obs)

        assert isinstance(obs, Z.PEChaserObs)
        return self._get_init_chaser_belief(obs)

    def _get_init_runner_belief(self, obs: Z.PERunnerObs) -> M.Belief:
        def init_belief_fn():
            return S.PEState(
                self._runner_start_loc,
                chaser_loc=self._chaser_start_loc,
                runner_dir=grid_lib.NORTH,
                chaser_dir=grid_lib.NORTH,
                runner_goal_loc=obs.goal,
                grid=self.grid
            )
        return M.InitialParticleBelief(init_belief_fn)   # type: ignore

    # pylint: disable=[unused-argument]
    def _get_init_chaser_belief(self, obs: Z.PEChaserObs) -> M.Belief:
        def init_belief_fn():
            runner_goal_locs = self.grid.get_runner_goal_locs(
                self._runner_start_loc
            )
            runner_goal_locs = list(runner_goal_locs)
            for loc in [self._runner_start_loc, self._chaser_start_loc]:
                if loc in runner_goal_locs:
                    runner_goal_locs.remove(loc)
            runner_goal_loc = random.choice(runner_goal_locs)

            return S.PEState(
                self._runner_start_loc,
                self._chaser_start_loc,
                runner_dir=grid_lib.NORTH,
                chaser_dir=grid_lib.NORTH,
                runner_goal_loc=runner_goal_loc,
                grid=self.grid
            )
        return M.InitialParticleBelief(init_belief_fn)   # type: ignore

    def step(self,
             state: M.State,
             action: M.JointAction
             ) -> Tuple[M.State, M.JointObservation, List[float], bool]:
        assert isinstance(state, S.PEState)
        next_state = self._get_next_state(state, action)
        obs, runner_detected = self._get_obs(next_state)
        reward = self._get_reward(next_state, runner_detected)
        done = self.is_terminal(next_state)
        return next_state, obs, reward, done

    def _get_next_state(self,
                        state: S.PEState,
                        action: M.JointAction) -> S.PEState:
        runner_a = action[self.RUNNER_IDX].action_num
        chaser_a = action[self.CHASER_IDX].action_num

        chaser_next_dir = A.ACTION_DIR_MAP[chaser_a]
        chaser_next_loc = self.get_agent_next_loc(chaser_a, state.chaser_loc)

        runner_next_loc = state.runner_loc
        runner_next_dir = A.ACTION_DIR_MAP[runner_a]
        if chaser_next_loc != state.runner_loc:
            runner_next_loc = self.get_agent_next_loc(
                runner_a, state.runner_loc
            )

        return S.PEState(
            runner_next_loc,
            chaser_next_loc,
            runner_dir=runner_next_dir,
            chaser_dir=chaser_next_dir,
            runner_goal_loc=state.runner_goal_loc,
            grid=state.grid
        )

    def get_agent_next_loc(self,
                           a_num: int,
                           loc: grid_lib.Loc) -> grid_lib.Loc:
        """Get next location for agent given action/dir """
        coords = self.grid.loc_to_coord(loc)
        new_coords = list(coords)
        if a_num == grid_lib.NORTH:
            new_coords[0] = max(0, coords[0]-1)
        elif a_num == grid_lib.SOUTH:
            new_coords[0] = min(self.grid.height-1, coords[0]+1)
        elif a_num == grid_lib.EAST:
            new_coords[1] = min(self.grid.width-1, coords[1]+1)
        elif a_num == grid_lib.WEST:
            new_coords[1] = max(0, coords[1]-1)
        new_loc = self.grid.coord_to_loc((new_coords[0], new_coords[1]))

        if new_loc in self.grid.block_locs:
            return loc
        return new_loc

    def _get_obs(self, s: S.PEState) -> Tuple[M.JointObservation, bool]:
        runner_obs = self._get_runner_obs(s)
        chaser_obs = self._get_chaser_obs(s)
        return M.JointObservation((runner_obs, chaser_obs)), chaser_obs.seen

    def _get_runner_obs(self, s: S.PEState) -> Z.PERunnerObs:
        r_walls, r_seen, r_heard = self._get_agent_obs(
            s.runner_loc, s.runner_dir, s.chaser_loc
        )
        return Z.PERunnerObs(
            walls=r_walls,
            seen=r_seen,
            heard=r_heard,
            goal=s.runner_goal_loc
        )

    def _get_chaser_obs(self, s: S.PEState) -> Z.PEChaserObs:
        c_walls, c_seen, c_heard = self._get_agent_obs(
            s.chaser_loc, s.chaser_dir, s.runner_loc
        )
        return Z.PEChaserObs(walls=c_walls, seen=c_seen, heard=c_heard)

    def _get_agent_obs(self,
                       agent_loc: grid_lib.Loc,
                       agent_dir: grid_lib.Dir,
                       opp_loc: grid_lib.Loc
                       ) -> Tuple[Z.WallObs, bool, bool]:
        walls = self._get_wall_obs(agent_loc)
        seen = self._get_opponent_seen(agent_loc, agent_dir, opp_loc)
        heard = self._get_opponent_heard(agent_loc, opp_loc)
        return walls, seen, heard

    def _get_wall_obs(self, loc: grid_lib.Loc) -> Z.WallObs:
        width, height = self.grid.width, self.grid.height
        adj_locs = [
            loc-width if loc >= width else -1,             # N
            loc+width if loc < (height-1)*width else -1,   # S
            loc+1 if loc % width < width-1 else -1,        # E
            loc-1 if loc % width > 0 else -1               # W
        ]

        adj_obs = [
            loc == -1 or loc in self.grid.block_locs for loc in adj_locs
        ]
        return tuple(adj_obs)   # type: ignore

    def _get_opponent_seen(self,
                           ego_loc: grid_lib.Loc,
                           ego_dir: grid_lib.Dir,
                           opp_loc: grid_lib.Loc) -> bool:
        fov = grid_lib.get_fov(ego_loc, ego_dir, self.grid)
        return opp_loc in fov

    def _get_opponent_heard(self,
                            ego_loc: grid_lib.Loc,
                            opp_loc: grid_lib.Loc) -> bool:
        return self.grid.manhattan_dist(ego_loc, opp_loc) <= self.HEARING_DIST

    def _get_reward(self, state: S.PEState, runner_seen: bool) -> List[float]:
        chaser_loc, runner_loc = state.chaser_loc, state.runner_loc
        if runner_loc == chaser_loc or runner_seen:
            return [-self.R_CAPTURE, self.R_CAPTURE]
        if runner_loc == state.runner_goal_loc:
            return [self.R_EVASION, -self.R_EVASION]
        return [self.R_RUNNER_ACTION, self.R_CHASER_ACTION]

    def is_terminal(self, state: M.State) -> bool:
        assert isinstance(state, S.PEState)
        if state.runner_loc == state.chaser_loc:
            return True
        if state.runner_loc == state.runner_goal_loc:
            return True
        return self._get_opponent_seen(
            state.chaser_loc, state.chaser_dir, state.runner_loc
        )

    def get_outcome(self, state: M.State) -> List:
        # Assuming this method is called on final timestep
        assert isinstance(state, S.PEState)
        chaser_loc, runner_loc = state.chaser_loc, state.runner_loc
        # check this first before relatively expensive detection check
        if runner_loc == chaser_loc:
            return [M.Outcomes.LOSS, M.Outcomes.WIN]
        if runner_loc == state.runner_goal_loc:
            return [M.Outcomes.WIN, M.Outcomes.LOSS]
        if self._get_opponent_seen(
            state.chaser_loc, state.chaser_dir, state.runner_loc
        ):
            return [M.Outcomes.LOSS, M.Outcomes.WIN]
        return [M.Outcomes.DRAW, M.Outcomes.DRAW]

    @property
    def runner_start_loc(self) -> grid_lib.Loc:
        """The start loc for the runner for the current episode """
        return self._runner_start_loc

    @property
    def chaser_start_loc(self) -> grid_lib.Loc:
        """The start loc for the chaser for the current episode """
        return self._chaser_start_loc

    @classmethod
    def get_args_parser(cls,
                        parser: Optional[ArgumentParser] = None
                        ) -> ArgumentParser:
        parser = super().get_args_parser(parser)
        parser.add_argument(
            "--grid_name", type=str, default='8by8',
            help="name of the grid to use env (default='8by8)"
        )
        return parser
