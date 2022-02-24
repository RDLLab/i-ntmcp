"""Actions for the Discrete Persuit Evasion Problem """
from typing import List

from intmcp.model import DiscreteAction, JointAction

from intmcp.envs.pe import grid as grid_lib

NORTH = 0
SOUTH = 1
EAST = 2
WEST = 3

ACTIONS = [NORTH, SOUTH, EAST, WEST]
ACTION_STRS = ["N", "S", "E", "W"]

# A map from each action to next direction
ACTION_DIR_MAP = {
    NORTH: grid_lib.NORTH,
    SOUTH: grid_lib.SOUTH,
    EAST: grid_lib.EAST,
    WEST: grid_lib.WEST,
}


class PEAction(DiscreteAction):
    """An action in the Runner Chaser Problem """

    def __str__(self):
        return ACTION_STRS[self.action_num]


def get_action_space(num_agents: int) -> List[JointAction]:
    """Get the problem action space """
    agent_action_spaces = []
    for _ in range(num_agents):
        agent_actions = []
        for action_num in ACTIONS:
            agent_actions.append(PEAction(action_num))
        agent_action_spaces.append(agent_actions)

    joint_action_space = JointAction.construct_joint_action_space(
        agent_action_spaces
    )
    return joint_action_space
