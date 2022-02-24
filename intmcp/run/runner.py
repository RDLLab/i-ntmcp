"""Module containing functions, etc for running INTMCP """
import time
import logging
from typing import Sequence, Optional, Iterable, Tuple

import intmcp.model as M
from intmcp.policy import BasePolicy
from intmcp.run import log as run_log
from intmcp.run import stats as stats_lib
from intmcp.run import render as render_lib


def run_episode_loop(env: M.POSGModel,
                     policies: Sequence[BasePolicy],
                     step_limit: int,
                     ) -> Iterable[Tuple[
                         M.POSGModel,
                         M.JointTimestep,
                         M.JointAction,
                         Sequence[BasePolicy],
                         bool
                     ]]:
    """Run policies in environment """
    assert len(policies) == env.num_agents

    state, joint_obs = env.reset()
    init_timestep = (state, joint_obs, [0.0] * env.num_agents, False)
    init_action = M.JointAction.get_joint_null_action(env.num_agents)
    yield env, init_timestep, init_action, policies, False

    episode_end = False
    steps = 0
    while not episode_end:
        agent_actions = []
        for i in range(env.num_agents):
            agent_actions.append(policies[i].step(joint_obs[i]))

        joint_action = M.JointAction(tuple(agent_actions))
        joint_timestep = env.step(state, joint_action)

        steps += 1
        state = joint_timestep[0]
        joint_obs = joint_timestep[1]
        episode_end = joint_timestep[3] or steps >= step_limit

        yield env, joint_timestep, joint_action, policies, episode_end


def run_sims(posg_model: M.POSGModel,
             policies: Sequence[BasePolicy],
             trackers: Iterable[stats_lib.Tracker],
             renderers: Iterable[render_lib.Renderer],
             num_episodes: int,
             step_limit: int,
             time_limit: Optional[int] = None,
             logger: Optional[logging.Logger] = None,
             **run_kwargs) -> stats_lib.AgentStatisticsMap:
    """Run INTMCP simulations """
    pause = run_kwargs.get("pause", False)
    render_asci = run_kwargs.get("render_asci", True)
    run_log.simulation_start(num_episodes, time_limit, logger)

    ep_num = 0
    time_limit_reached = False
    run_start_time = time.time()

    for tracker in trackers:
        tracker.reset()

    while ep_num < num_episodes and not time_limit_reached:
        run_log.episode_start(ep_num, logger)

        for tracker in trackers:
            tracker.reset_episode()

        for policy in policies:
            policy.reset()

        timestep_sequence = run_episode_loop(posg_model, policies, step_limit)
        for t, (env, timestep, a, pis, ep_end) in enumerate(timestep_sequence):
            if t == 0:
                run_log.initial_timestep(timestep, logger, render_asci)
                if pause:
                    input("...")
                continue

            run_log.joint_step(t, a, timestep, logger, render_asci)
            if pause:
                input("...")

            for tracker in trackers:
                tracker.step(env, timestep, a, pis, ep_end)

            render_lib.generate_renders(
                renderers, env, timestep, a, pis, ep_end
            )

        run_log.episode_end(
            ep_num, stats_lib.generate_episode_statistics(trackers), logger
        )

        run_log.progress(ep_num, num_episodes, logger)
        ep_num += 1

        if pause:
            input()

        if time_limit is not None and time.time()-run_start_time > time_limit:
            time_limit_reached = True
            run_log.time_limit_reached(ep_num, time_limit, logger)

    statistics = stats_lib.generate_statistics(trackers)
    run_log.simulation_end(statistics, logger)

    return statistics
