"""Module for generating .ipomdp files for the Runner Chaser env.

Assumes:
- two agent environment
- number of actions are equal for both agents
- number of observation are equal for both agents


.ipomdp format description
--------------------------
L1 |S| |A_i| |O_i|

L3 Initial Belief

L5 |S| s0 b0(s0) s1 b0(s1) ... s|S| b0(s|S|)

L7 Transitions

L9 s0

L11 a0 a0 |S| s0 T(s0|<a0, a0>, s0) s1 T(s1|<a0, a0>, s0) ...
L12 a0 a1 ...

... Repeat for all joint actions, then repeat blocks with s1, ..., s|S|


"""
from itertools import product
from typing import List, Dict, Tuple

import intmcp.model as M

from intmcp.envs.rc import obs as Z
from intmcp.envs.rc import state as S
from intmcp.envs.rc import action as A
from intmcp.envs.rc.model import RCModel
from intmcp.envs.rc import grid as grid_lib
from intmcp.envs.rc import ipl_model_write as write_lib

# pylint: disable=[protected-access]


class RCStateWithAbsorbing(S.RCState):
    """Adds an absorbing flag to RC State """

    def __init__(self,
                 runner_loc: grid_lib.Loc,
                 chaser_loc: grid_lib.Loc,
                 absorbing: bool,
                 grid: grid_lib.Grid):
        super().__init__(runner_loc, chaser_loc, grid)
        self.absorbing = absorbing

    def __hash__(self):
        return hash((self.runner_loc, self.chaser_loc, self.absorbing))

    def __eq__(self, other):
        return (
            self.runner_loc == other.runner_loc
            and self.chaser_loc == other.chaser_loc
            and self.absorbing == other.absorbing
        )


class RCObsWithAbsorbing(Z.RCObs):
    """Adds an absorbing obs to RC Obs """

    def __init__(self,
                 loc: grid_lib.Loc,
                 adj_obs: Tuple[int, ...],
                 absorbing: bool):
        super().__init__(loc, adj_obs)
        self.absorbing = absorbing

    def __hash__(self):
        return hash((self.loc, self.adj_obs, self.absorbing))

    def __eq__(self, other):
        return (
            (self.loc, self.adj_obs, self.absorbing)
            == (other.loc, other.adj_obs, other.absorbing)
        )


def _get_state_space(model: RCModel, use_absorbing: bool) -> List[S.RCState]:
    all_locs = model.grid.get_all_unblocked_locs()

    state_space: List[S.RCState] = []
    if use_absorbing:
        state_space = [
            RCStateWithAbsorbing(r_loc, c_loc, ab, model.grid)
            for r_loc, c_loc, ab in product(all_locs, all_locs, [True, False])
        ]
    else:
        state_space = [
            S.RCState(r_loc, c_loc, model.grid)
            for r_loc, c_loc in product(all_locs, repeat=2)
        ]

    return state_space


def _get_action_space(model: RCModel
                      ) -> Tuple[
                          List[A.JointAction], write_lib.ActionMap, int
                      ]:
    action_space = model.action_space

    num_agent_actions = 0
    for i in range(model.NUM_AGENTS):
        agent_actions = model.get_agent_action_space(i)
        if num_agent_actions == 0:
            num_agent_actions = len(agent_actions)
        else:
            assert len(agent_actions) == num_agent_actions

    action_map: Dict[M.JointAction, Tuple[int, int]] = {}
    for a in action_space:
        r_a = a.get_agent_action(model.RUNNER_IDX)
        c_a = a.get_agent_action(model.CHASER_IDX)
        assert isinstance(r_a, A.RCAction) and isinstance(c_a, A.RCAction)
        action_map[a] = (r_a.action_num, c_a.action_num)

    return action_space, action_map, num_agent_actions


def _get_obs_space(model: RCModel,
                   use_absorbing: bool
                   ) -> Tuple[List[Z.RCObs], List[Z.RCObs], int]:
    all_adj_obs = list(product(Z.CELL_OBS, repeat=4))

    runner_start_locs = model.grid.init_runner_locs
    chaser_start_locs = model.grid.init_chaser_locs
    assert len(runner_start_locs) == len(chaser_start_locs)

    runner_obs: List[Z.RCObs] = []
    chaser_obs: List[Z.RCObs] = []
    if use_absorbing:
        runner_obs = [
            RCObsWithAbsorbing(loc, adj_obs, ab)
            for loc, adj_obs, ab in product(
                    runner_start_locs, all_adj_obs, [True, False]
            )
        ]
        chaser_obs = [
            RCObsWithAbsorbing(loc, adj_obs, ab)
            for loc, adj_obs, ab in product(
                    chaser_start_locs, all_adj_obs, [True, False]
            )
        ]
    else:
        runner_obs = [
            Z.RCObs(loc, adj_obs)
            for loc, adj_obs in product(runner_start_locs, all_adj_obs)
        ]
        chaser_obs = [
            Z.RCObs(loc, adj_obs)
            for loc, adj_obs in product(chaser_start_locs, all_adj_obs)
        ]
    return runner_obs, chaser_obs, len(runner_obs)


def _get_b0_dist(model: RCModel,
                 state_space: List[S.RCState],
                 use_absorbing: bool) -> List[float]:
    r_start_locs = model.grid.init_runner_locs
    c_start_locs = model.grid.init_chaser_locs

    p = 1.0 / (len(r_start_locs)*len(c_start_locs))

    b0_dist: List[float] = []
    for s in state_space:
        if s.runner_loc in r_start_locs and s.chaser_loc in c_start_locs:
            if use_absorbing:
                assert isinstance(s, RCStateWithAbsorbing)
                if s.absorbing:
                    b = 0.0
                else:
                    b = p
            else:
                b = p
        else:
            b = 0.0
        b0_dist.append(b)

    return b0_dist


def _get_tfn(model: RCModel,
             state_space: List[S.RCState],
             action_space: List[M.JointAction],
             use_absorbing: bool) -> write_lib.Tfn:
    t_fn: write_lib.Tfn = {}

    for s in state_space:
        assert isinstance(s, S.RCState)
        t_fn[s] = {}

        for a in action_space:
            true_next_s = model._get_next_state(s, a)

            if use_absorbing:
                assert isinstance(s, RCStateWithAbsorbing)
                s_terminal = model.is_terminal(s)
                if s_terminal and s.absorbing:
                    true_next_s = s
                elif s_terminal:
                    true_next_s = RCStateWithAbsorbing(
                        runner_loc=s.runner_loc,
                        chaser_loc=s.chaser_loc,
                        absorbing=True,
                        grid=s.grid
                    )
                elif model.is_terminal(true_next_s):
                    true_next_s = RCStateWithAbsorbing(
                        runner_loc=true_next_s.runner_loc,
                        chaser_loc=true_next_s.chaser_loc,
                        absorbing=True,
                        grid=s.grid
                    )
                else:
                    true_next_s = RCStateWithAbsorbing(
                        runner_loc=true_next_s.runner_loc,
                        chaser_loc=true_next_s.chaser_loc,
                        absorbing=False,
                        grid=s.grid
                    )

            next_s_probs = []
            for next_s in state_space:
                if next_s == true_next_s:
                    next_s_probs.append(1.0)
                else:
                    next_s_probs.append(0.0)

            t_fn[s][a] = next_s_probs

    return t_fn


def _get_zfns(model: RCModel,
              state_space: List[S.RCState],
              action_space: List[M.JointAction],
              runner_obs: List[Z.RCObs],
              chaser_obs: List[Z.RCObs],
              use_absorbing: bool
              ) -> Tuple[write_lib.Zfn, write_lib.Zfn]:
    r_zfn: write_lib.Zfn = {}
    c_zfn: write_lib.Zfn = {}

    for s in state_space:
        assert isinstance(s, S.RCState)
        r_zfn[s] = {}
        c_zfn[s] = {}

        s_terminal = model.is_terminal(s)

        for a in action_space:
            # pylint: disable=[protected-access]
            true_obs = model._get_obs(s)
            true_r_obs = true_obs.get_agent_obs(model.RUNNER_IDX)
            true_c_obs = true_obs.get_agent_obs(model.CHASER_IDX)

            if use_absorbing:
                assert isinstance(true_r_obs, Z.RCObs)
                assert isinstance(true_c_obs, Z.RCObs)
                true_r_obs = RCObsWithAbsorbing(
                    loc=true_r_obs.loc,
                    adj_obs=true_r_obs.adj_obs,
                    absorbing=s_terminal
                )
                true_c_obs = RCObsWithAbsorbing(
                    loc=true_c_obs.loc,
                    adj_obs=true_c_obs.adj_obs,
                    absorbing=s_terminal
                )

            r_obs_probs = [float(r_obs == true_r_obs) for r_obs in runner_obs]
            c_obs_probs = [float(c_obs == true_c_obs) for c_obs in chaser_obs]

            r_zfn[s][a] = r_obs_probs
            c_zfn[s][a] = c_obs_probs

    return r_zfn, c_zfn


def _get_rfns(model: RCModel,
              state_space: List[S.RCState],
              action_space: List[M.JointAction],
              use_absorbing: bool
              ) -> Tuple[write_lib.Rfn, write_lib.Rfn]:
    r_rfn: write_lib.Rfn = {}
    c_rfn: write_lib.Rfn = {}

    for s in state_space:
        r_rfn[s] = {}
        c_rfn[s] = {}

        for a in action_space:
            # pylint: disable=[protected-access]
            next_s = model._get_next_state(s, a)
            rewards = model._get_reward(next_s)

            if use_absorbing:
                assert isinstance(s, RCStateWithAbsorbing)
                if s.absorbing:
                    rewards = [0.0, 0.0]

            r_rfn[s][a] = rewards[model.RUNNER_IDX]
            c_rfn[s][a] = rewards[model.CHASER_IDX]

    return r_rfn, c_rfn


def gen_ipomdp(model: RCModel, use_absorbing: bool) -> write_lib.ProblemDef:
    """Generate IPOMDP problem definition for RC model """
    print("Generating IPOMDP definition from model")

    print("Generating state space")
    state_space = _get_state_space(model, use_absorbing)
    num_states = len(state_space)
    print(f"  state space with {num_states} states generated.")

    print("Generating action space")
    action_space, action_map, num_agent_actions = _get_action_space(model)
    print(
        f"  action space with {len(action_space)} joint actions and "
        f"{num_agent_actions} agent actions generated."
    )

    print("Generating obs space")
    runner_obs, chaser_obs, num_agent_obs = _get_obs_space(
        model, use_absorbing
    )
    print(f"  runner obs space generated with {len(runner_obs)} obs.")
    print(f"  chaser obs space generated with {len(chaser_obs)} obs.")

    print("Generating initial belief dist")
    b0_dist = _get_b0_dist(model, state_space, use_absorbing)
    print(f"  Initial belief dist generated with {len(b0_dist)} dimensions")

    print("Generating transition function")
    t_fn = _get_tfn(model, state_space, action_space, use_absorbing)

    print("Generating observation functions")
    z_fns = _get_zfns(
        model, state_space, action_space, runner_obs, chaser_obs, use_absorbing
    )

    print("Generating reward functions")
    r_fns = _get_rfns(model, state_space, action_space, use_absorbing)

    print("Generation of IPOMDP definition complete")
    return write_lib.ProblemDef(
        num_states=num_states,
        state_space=state_space,
        num_agent_actions=num_agent_actions,
        action_space=action_space,
        action_map=action_map,
        num_agent_obs=num_agent_obs,
        obs_spaces=(runner_obs, chaser_obs),
        b0_dist=b0_dist,
        t_fn=t_fn,
        z_fns=z_fns,
        r_fns=r_fns
    )


if __name__ == "__main__":
    parser = RCModel.get_args_parser()
    parser.add_argument(
        "output_file", type=str,
        help=".ipomdp file to output too"
    )
    parser.add_argument(
        "--use_absorbing", action='store_true',
        help="Model IPOMDP with absorbing states"
    )
    parser.add_argument(
        "--include_b0", action='store_true',
        help="Include initial belief in .ipomdp file"
    )
    args = parser.parse_args()
    rc_model = RCModel(**vars(args))

    rc_problem_def = gen_ipomdp(rc_model, args.use_absorbing)
    write_lib.write_ipomdp(
        rc_problem_def, args.output_file, args.include_b0
    )
