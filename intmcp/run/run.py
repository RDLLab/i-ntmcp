"""Main run function for running INTMCP on a model """
import random
from typing import Type, Optional, Dict, List
from argparse import ArgumentParser, Namespace

import numpy as np

import intmcp.model as M

from intmcp import policy
import intmcp.log as log_lib
from intmcp.run.runner import run_sims
from intmcp.run import stats as stats_lib
from intmcp.run import render as render_lib


def get_run_args_parser(parser: Optional[ArgumentParser] = None
                        ) -> ArgumentParser:
    """Get ArgumentParser containing all runner arguments """
    if parser is None:
        parser = ArgumentParser(conflict_handler='resolve')
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed (default=None)"
    )
    parser.add_argument(
        "-s", "--step_limit", type=int, default=20,
        help="Step limit (default=20)"
    )
    parser.add_argument(
        "-e", "--num_episodes", type=int, default=100,
        help="Number of episodes (default=100)"
    )
    parser.add_argument(
        "--time_limit", type=int, default=None,
        help="Time limit (in seconds) for run (default=None)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.95,
        help="Discount for run (default=0.95)"
    )
    parser.add_argument(
        "-st", "--show_tree", type=int, default=None,
        help="Display search tree up to given depth each step (default=None)"
    )
    parser.add_argument(
        "-sp", "--show_pi", type=int, default=None,
        help="Display policies up to given depth each step (default=None)"
    )
    parser.add_argument(
        "-sb", "--show_belief", type=int, default=None,
        help=(
            "Display beliefs up to given level each step, set this to -1 to"
            " display up to policy nesting level  (default=None)"
        )
    )
    parser.add_argument(
        "-l", "--log_level", type=int, default=log_lib.INFO1,
        help="Logging level, (default=loging.INFO-1)"
    )
    parser.add_argument(
        "--pause", action="store_true",
        help="Pause for user input between each step"
    )
    return parser


def parse_run_kwargs(args: Namespace) -> dict:
    """Parse parser and extract runner kwargs """
    run_kwargs = {
        "seed": args.seed,
        "num_episodes": args.num_episodes,
        "step_limit": args.step_limit,
        "time_limit": args.time_limit,
        "show_tree": args.show_tree,
        "show_pi": args.show_pi,
        "show_belief": args.show_belief,
        "log_level": args.log_level,
        "gamma": args.gamma,
        "pause": args.pause
    }

    log_lib.config_logger(level=args.log_level)

    return run_kwargs


def main_run(model_class: Type[M.POSGModel],
             policy_classes: List[Type[policy.BasePolicy]],
             extra_model_kwargs: Optional[Dict] = None,
             extra_policy_kwargs_list: Optional[List[Dict]] = None,
             parser: Optional[ArgumentParser] = None):
    """ Run Model class with given policies classes, handling parsing of cmd
    line args.
    """
    parser = get_run_args_parser(parser)
    parser = model_class.get_args_parser(parser)
    for pi_class in policy_classes:
        parser = pi_class.get_args_parser(parser)
    args = parser.parse_args()

    return main_run_with_args(
        model_class,
        policy_classes,
        args,
        extra_model_kwargs=extra_model_kwargs,
        extra_policy_kwargs_list=extra_policy_kwargs_list
    )


def main_run_with_args(model_class: Type[M.POSGModel],
                       policy_classes: List[Type[policy.BasePolicy]],
                       args: Namespace,
                       extra_model_kwargs: Optional[Dict] = None,
                       extra_policy_kwargs_list: Optional[List[Dict]] = None
                       ) -> None:
    """ Run Model class with given policies classes, args and kwargs """
    run_kwargs = parse_run_kwargs(args)

    if extra_model_kwargs is None:
        extra_model_kwargs = {}

    if extra_policy_kwargs_list is None:
        extra_policy_kwargs_list = [{} for i in range(len(policy_classes))]

    if run_kwargs["seed"] is not None:
        random.seed(run_kwargs["seed"])
        np.random.seed(run_kwargs["seed"])

    model_kwargs = {**vars(args)}
    model_kwargs.update(extra_model_kwargs)
    model = model_class(**model_kwargs)

    num_agents = model.num_agents
    assert len(policy_classes) == num_agents
    assert len(extra_policy_kwargs_list) == num_agents
    policies: List[policy.BasePolicy] = []

    for i in range(num_agents):
        pi_class = policy_classes[i]
        kwargs = {**vars(args)}
        kwargs.update(extra_policy_kwargs_list[i])
        policies.append(
            pi_class.initialize(model=model, ego_agent=i, **kwargs)
        )

    trackers = stats_lib.get_default_trackers(args.gamma, policies)
    renderers = render_lib.get_renderers(
        show_pi=args.show_pi,
        show_belief=args.show_belief,
        show_tree=args.show_tree
    )

    run_sims(model, policies, trackers, renderers, **run_kwargs)
