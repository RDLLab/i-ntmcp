"""Module for writing an .ipomdp file for the Runner Chaser env.

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
from typing import NamedTuple, List, Dict, Tuple

import intmcp.model as M

from intmcp.envs.rc import obs as Z
from intmcp.envs.rc import state as S

Tfn = Dict[S.RCState, Dict[M.JointAction, List[float]]]
Zfn = Dict[S.RCState, Dict[M.JointAction, List[float]]]
Rfn = Dict[S.RCState, Dict[M.JointAction, float]]
B0Dist = List[float]

ActionMap = Dict[M.JointAction, Tuple[int, int]]

R_IDX = 0
C_IDX = 1


class ProblemDef(NamedTuple):
    """problem definition for writing .ipomdp file """
    num_states: int
    state_space: List[S.RCState]
    num_agent_actions: int
    action_space: List[M.JointAction]
    action_map: ActionMap
    num_agent_obs: int
    obs_spaces: Tuple[List[Z.RCObs], List[Z.RCObs]]
    b0_dist: B0Dist
    t_fn: Tfn
    z_fns: Tuple[Zfn, Zfn]
    r_fns: Tuple[Rfn, Rfn]


def _write_problem_size(pdef: ProblemDef, fout):
    fout.write(
        f"{pdef.num_states} {pdef.num_agent_actions} {pdef.num_agent_obs}\n"
    )
    fout.write("\n")


def _write_initial_belief(pdef: ProblemDef, fout):
    fout.write("Initial Belief\n")
    fout.write("\n")

    output = [f"{pdef.num_states}"]
    for snum, b in enumerate(pdef.b0_dist):
        output.append(f"{snum} {b:.6f}")
    output_str = " ".join(output)

    fout.write(f"{output_str}\n")
    fout.write("\n")


def _write_transition(pdef: ProblemDef, fout):
    fout.write("Transitions\n")
    fout.write("\n")

    for snum, s in enumerate(pdef.state_space):
        assert isinstance(s, S.RCState)
        fout.write(f"{snum}\n")
        fout.write("\n")

        for a_joint, (a_i, a_j) in pdef.action_map.items():
            fout.write(f"{a_i} {a_j} {pdef.num_states}")
            next_s_probs = pdef.t_fn[s][a_joint]

            for s_next_num, prob in enumerate(next_s_probs):
                fout.write(f" {s_next_num} {prob:.6f}")

            fout.write("\n")

        fout.write("\n")


def _write_agent_observation(agent_id: int, pdef: ProblemDef, fout):
    fout.write(f"Observation Agent {agent_id}\n")
    fout.write("\n")

    for snum, s in enumerate(pdef.state_space):
        assert isinstance(s, S.RCState)
        fout.write(f"{snum}\n")
        fout.write("\n")

        for a_joint, (a_i, a_j) in pdef.action_map.items():
            fout.write(f"{a_i} {a_j} {pdef.num_agent_obs}")
            obs_probs = pdef.z_fns[agent_id][s][a_joint]

            for o_num, prob in enumerate(obs_probs):
                fout.write(f" {o_num} {prob:.6f}")

            fout.write("\n")

        fout.write("\n")


def _write_observations(pdef: ProblemDef, fout):
    _write_agent_observation(R_IDX, pdef, fout)
    _write_agent_observation(C_IDX, pdef, fout)


def _write_agent_reward(agent_id: int, pdef: ProblemDef, fout):
    fout.write(f"Rewards Agent {agent_id}\n")
    fout.write("\n")

    num_joint_actions = len(pdef.action_space)
    for snum, s in enumerate(pdef.state_space):
        assert isinstance(s, S.RCState)
        fout.write(f"{snum} {num_joint_actions}\n")
        fout.write("\n")

        for a_joint, (a_i, a_j) in pdef.action_map.items():
            rew = pdef.r_fns[agent_id][s][a_joint]
            fout.write(f" {a_i} {a_j} {rew:.6f}")

        fout.write("\n")


def _write_rewards(pdef: ProblemDef, fout):
    _write_agent_reward(R_IDX, pdef, fout)
    _write_agent_reward(C_IDX, pdef, fout)


def write_ipomdp(pdef: ProblemDef, output_file: str, include_b0: bool):
    """Write problem to file in .ipomdp format """
    print(f"Writing .ipomdp file to {output_file}")

    with open(output_file, "w") as fout:
        print("Writing problem size")
        _write_problem_size(pdef, fout)

        if include_b0:
            print("Writing initial belief")
            _write_initial_belief(pdef, fout)

        print("Writing transition function")
        _write_transition(pdef, fout)

        print("Writing observation functions")
        _write_observations(pdef, fout)

        print("Writing reward functions")
        _write_rewards(pdef, fout)
