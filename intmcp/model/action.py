"""Base action class """
from typing import Tuple


class Action:
    """An abstract action class """

    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def get_null_action() -> 'Action':
        """Get NullAction """
        return NullAction()


class DiscreteAction(Action):
    """A simple discrete action class """

    def __init__(self, action_num: int):
        self.action_num = action_num

    def __str__(self):
        return str(self.action_num)

    def __hash__(self):
        return self.action_num

    def __eq__(self, other):
        return self.action_num == other.action_num


class NullAction(Action):
    """An object for representing a None action.

    Useful for when dealing with histories.
    """

    def __hash__(self):
        return hash(None)

    def __eq__(self, other):
        return isinstance(other, NullAction)

    def __str__(self):
        return "NullAction"

    def __repr__(self):
        return self.__str__()


class JointAction(Action):
    """An joint action class

    Attributes
    ----------
    action_tuple : Tuple[Action]
        tuple of Actions
    """

    def __init__(self, action_tuple: Tuple[Action, ...]):
        self.action_tuple = action_tuple

    def contains_action(self, action, agent_idx):
        """Check if joint action contains given action for given agent """
        return self.action_tuple[agent_idx] == action

    def get_agent_action(self, agent_idx):
        """Get the action for given agent """
        return self.action_tuple[agent_idx]

    def __hash__(self):
        return hash(self.action_tuple)

    def __eq__(self, other):
        return self.action_tuple == other.action_tuple

    def __getitem__(self, key):
        return self.action_tuple[key]

    def __iter__(self):
        return JointActionIterator(self)

    @classmethod
    def get_joint_null_action(cls, num_agents: int) -> 'JointAction':
        """Get Joint Null Action """
        agent_actions = [Action.get_null_action() for _ in range(num_agents)]
        return cls(tuple(agent_actions))

    @staticmethod
    def construct_joint_action_space(agent_action_spaces):
        """Construct joint action space from set of agent action spaces """
        joint_action_lists = JointAction._add_next_actions(
            0, agent_action_spaces
        )
        joint_action_space = []
        for action_list in joint_action_lists:
            joint_action_space.append(
                JointAction(tuple(action_list))
            )
        return joint_action_space

    @staticmethod
    def get_agent_action_space(joint_action_space, agent_id):
        """Get the action space of an agent from the joint action space """
        agent_action_space = set()
        for joint_action in joint_action_space:
            agent_action = joint_action.get_agent_action(agent_id)
            agent_action_space.add(agent_action)
        return list(agent_action_space)

    @staticmethod
    def _add_next_actions(agent_idx, agent_action_spaces):
        if agent_idx == len(agent_action_spaces)-1:
            joint_action_space = []
            for action in agent_action_spaces[agent_idx]:
                joint_action_space.append([action])
            return joint_action_space

        joint_action_space = []
        for action in agent_action_spaces[agent_idx]:
            sub_joint_actions = JointAction._add_next_actions(
                agent_idx + 1, agent_action_spaces
            )
            for sub_actions in sub_joint_actions:
                joint_action_space.append(
                    [action] + sub_actions
                )
        return joint_action_space

    def __str__(self):
        output = [str(a) for a in self.action_tuple]
        return f"({','.join(output)})"


class JointActionIterator:
    """Iterator for the JointAction class """

    def __init__(self, joint_action: JointAction):
        self.joint_action = joint_action
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx < len(self.joint_action.action_tuple):
            self._idx += 1
            return self.joint_action[self._idx-1]
        raise StopIteration
