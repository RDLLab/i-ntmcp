"""Actions for Runner Chaser Problem """
from typing import List

from intmcp.model import DiscreteAction, JointAction

from intmcp.envs.rc import grid as grid_lib


class RCAction(DiscreteAction):
    """An action in the Runner Chaser Problem """

    def __str__(self):
        return grid_lib.DIR_STRS[self.action_num]


def get_action_space(num_agents: int) -> List[JointAction]:
    """Get the problem action space """
    agent_action_spaces = []
    for _ in range(num_agents):
        agent_actions = []
        for action_num in grid_lib.DIRS:
            agent_actions.append(RCAction(action_num))
        agent_action_spaces.append(agent_actions)

    joint_action_space = JointAction.construct_joint_action_space(
        agent_action_spaces
    )
    return joint_action_space
