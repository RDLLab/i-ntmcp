"""Get the reward bound for given tree search algorithm and problem

As per the original POMCP paper:

Reward upper is the highest return achieved during sample runs of the
Tree search with no exploration (i.e. UCT c param = 0)

Reward lower is the lowest return achieved during random sample rollouts (or
rollouts using rollout policy)
"""
import sys
from argparse import ArgumentParser

from intmcp.run import run, Result
from intmcp.tree import SEARCH_TREES
from intmcp.policy import POLICIES, RandomPolicy
from intmcp.envs import ENVS, get_available_env_policies


if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')
    parser.add_argument("env", type=str, help="Name of environment")
    parser.add_argument("test_agent", type=int,
                        help="ID of agent to get bounds for.")
    parser.add_argument("policies", type=str, nargs="*",
                        help="The name of policies/trees to use.")

    if len(sys.argv) < 4:
        # parser will hit error and display help menu
        args = parser.parse_args()

    env_name = sys.argv[1]
    test_agent = int(sys.argv[2])
    policy_names = []
    for argv in sys.argv[3:]:
        if argv.startswith("-"):
            break
        policy_names.append(argv)

    print(env_name)
    print(test_agent)
    print(policy_names)

    if env_name not in ENVS:
        print(
            f"Unknown environment name '{env_name}'. Please choose one of: "
            f"{list(ENVS)}."
        )
        sys.exit()

    env_class = ENVS[env_name]
    env_policies = get_available_env_policies(env_name)

    policy_classes = []
    policy_class_kwargs = [{} for _ in policy_names]   # type: ignore
    for i, pi_name in enumerate(policy_names):
        if pi_name in SEARCH_TREES:
            policy_classes.append(SEARCH_TREES[pi_name])
            if i == test_agent:
                policy_class_kwargs[i] = {"uct_c": 0.0}
        elif pi_name in POLICIES:
            policy_classes.append(POLICIES[pi_name])
        elif pi_name in env_policies:
            policy_classes.append(env_policies[pi_name])
        else:
            print(
                f"Unknown policy/tree name '{pi_name}'. Please choose one of: "
                f"{[*SEARCH_TREES, *POLICIES, *env_policies]}."
            )
            sys.exit()

    parser = run.get_run_args_parser(parser)
    parser = env_class.get_args_parser(parser)    # type: ignore
    for pi_class in policy_classes:
        parser = pi_class.get_args_parser(parser)
    args = parser.parse_args()

    upper_result = run.main_run_with_args(
        env_class,
        policy_classes,
        args,
        extra_policy_kwargs_list=policy_class_kwargs,
    )

    policy_classes[test_agent] = RandomPolicy
    policy_class_kwargs[test_agent] = {}
    lower_result = run.main_run_with_args(
        env_class,
        policy_classes,
        args,
        extra_policy_kwargs_list=policy_class_kwargs,
    )

    test_agent_uresult = upper_result.results[test_agent]
    test_agent_lresult = lower_result.results[test_agent]

    upper_bound = test_agent_uresult["episode_discounted_returns_max"]
    lower_bound = test_agent_lresult["episode_discounted_returns_min"]
    diff = upper_bound - lower_bound
    print(f"Upper bound = {Result.get_str_repr(upper_bound)}")
    print(f"Lower bound = {Result.get_str_repr(lower_bound)}")
    print(f"Difference = {Result.get_str_repr(diff)}")
