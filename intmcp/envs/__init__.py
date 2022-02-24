"""A collection of POSG Environment/Problem Implementations """
from typing import Dict, Type

import intmcp.model as M
from intmcp import tree as tree_lib
from intmcp import policy as policy_lib

from intmcp.envs import pe
from intmcp.envs import rc


ENVS: Dict[str, Type[M.POSGModel]] = {
    "pe": pe.PEModel,
    "rc": rc.RCModel,
}


def get_env_class(env_name: str) -> Type[M.POSGModel]:
    """Get class for given env name """
    if env_name not in ENVS:
        raise ValueError(
            f"Unknown environment '{env_name}'. Please choose one of: "
            f"{list(ENVS)}."
        )
    return ENVS[env_name]


def get_available_env_policies(env_name: str) -> Dict:
    """Get set of available policies for a given environment """
    assert env_name in ENVS
    env_module = eval(env_name)    # pylint: disable=[eval-used]
    if hasattr(env_module, "POLICIES"):
        return env_module.POLICIES
    return {}


def get_policy_class(env_name: str,
                     policy_name: str) -> Type[policy_lib.BasePolicy]:
    """Get the policy classs for given env and name """
    if policy_name in tree_lib.SEARCH_TREES:
        return tree_lib.SEARCH_TREES[policy_name]

    if policy_name in policy_lib.POLICIES:
        return policy_lib.POLICIES[policy_name]

    env_policies = get_available_env_policies(env_name)
    if policy_name in env_policies:
        return env_policies[policy_name]

    raise ValueError(
        f"Unknown policy/tree name '{policy_name}'. Please choose one of: "
        f"{[*tree_lib.SEARCH_TREES, *policy_lib.POLICIES, *env_policies]}."
    )
