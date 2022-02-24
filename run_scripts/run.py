"""Main run script for running examples """
import sys
from argparse import ArgumentParser

from intmcp.run import run
from intmcp import envs as env_lib


if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')
    parser.add_argument("env", type=str, help="Name of environment to run")
    parser.add_argument("policies", type=str, nargs="*",
                        help="The name of policies/trees to use.")
    if len(sys.argv) < 3:
        # parser will hit error and display help menu
        args = parser.parse_args()

    env_name = sys.argv[1]
    policy_names = []
    for argv in sys.argv[2:]:
        if argv.startswith("-"):
            break
        policy_names.append(argv)

    env_class = env_lib.get_env_class(env_name)
    env_policies = env_lib.get_available_env_policies(env_name)

    policy_classes = []
    for policy_name in policy_names:
        policy_class = env_lib.get_policy_class(env_name, policy_name)
        policy_classes.append(policy_class)

    run.main_run(env_class, policy_classes, parser=parser)
