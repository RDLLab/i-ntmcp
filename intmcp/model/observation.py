"""The base observation class """
from typing import Tuple, List


class Observation:
    """An abstract observation """

    def render_asci(self):
        """Get ascii repr of the observation """
        return self.__str__()

    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def get_null_obs() -> 'Observation':
        """Get NullAction """
        return NullObservation()


class DiscreteObservation(Observation):
    """A simple discrete observation """

    def __init__(self, obs_num: int):
        self.obs_num = obs_num

    def __str__(self):
        return str(self.obs_num)

    def __hash__(self):
        return int(self.obs_num)

    def __eq__(self, other):
        return int(self.obs_num) == int(other.obs_num)


class NullObservation(Observation):
    """An object for representing a None observation.

    Useful for when dealing with histories.
    """

    def __hash__(self):
        return hash(None)

    def __eq__(self, other):
        return isinstance(other, NullObservation)

    def __str__(self):
        return "NullObservation"

    def __repr__(self):
        return self.__str__()


class JointObservation(Observation):
    """An joint observation class

    Attributes
    ----------
    obs_tuple : Tuple[Observation]
        tuple of Observations
    """

    def __init__(self, obs_tuple: Tuple[Observation, ...]):
        self.obs_tuple = obs_tuple

    def contains_obs(self, obs: Observation, agent_idx: int) -> bool:
        """Check if joint obs contains given obs for given agent """
        return self.obs_tuple[agent_idx] == obs

    def get_agent_obs(self, agent_idx: int) -> Observation:
        """Get obs for given agent """
        return self.obs_tuple[agent_idx]

    @staticmethod
    def get_agent_obs_space(joint_obs_space: List['JointObservation'],
                            agent_id: int) -> List[Observation]:
        """Get the obs space of an agent from the joint obs space """
        agent_obs_space = set()
        for joint_obs in joint_obs_space:
            agent_obs = joint_obs.get_agent_obs(agent_id)
            agent_obs_space.add(agent_obs)
        return list(agent_obs_space)

    @classmethod
    def get_joint_null_obs(cls, num_agents: int) -> 'JointObservation':
        """Get Joint Null Observation """
        agent_obs = [Observation.get_null_obs() for _ in range(num_agents)]
        return cls(tuple(agent_obs))

    @staticmethod
    def construct_joint_obs_space(agent_obs_spaces):
        """Construct joint obs space from set of agent obs spaces """
        joint_obs_lists = JointObservation._add_next_obs(
            0, agent_obs_spaces
        )
        joint_obs_space = []
        for obs_list in joint_obs_lists:
            joint_obs_space.append(
                JointObservation(tuple(obs_list))
            )
        return joint_obs_space

    @staticmethod
    def _add_next_obs(agent_idx, agent_obs_spaces):
        if agent_idx == len(agent_obs_spaces)-1:
            joint_obs_space = []
            for obs in agent_obs_spaces[agent_idx]:
                joint_obs_space.append([obs])
            return joint_obs_space

        joint_obs_space = []
        for obs in agent_obs_spaces[agent_idx]:
            sub_joint_obs = JointObservation._add_next_obs(
                agent_idx + 1, agent_obs_spaces
            )
            for sub_obs in sub_joint_obs:
                joint_obs_space.append(
                    [obs] + sub_obs
                )
        return joint_obs_space

    def __hash__(self):
        return hash(self.obs_tuple)

    def __eq__(self, other):
        return self.obs_tuple == other.obs_tuple

    def __getitem__(self, key):
        return self.obs_tuple[key]

    def __iter__(self):
        return JointObservationIterator(self)

    def __str__(self):
        output = [str(o) for o in self.obs_tuple]
        return f"({','.join(output)})"

    def render_asci(self):
        output = [o.render_asci() for o in self.obs_tuple]
        if any(o.count("\n") > 1 for o in output):
            return "\n".join(output)
        return f"({','.join(output)})"


class JointObservationIterator:
    """Iterator for the JointObservation class """

    def __init__(self, joint_obs: JointObservation):
        self.joint_obs = joint_obs
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx < len(self.joint_obs.obs_tuple):
            self._idx += 1
            return self.joint_obs[self._idx-1]
        raise StopIteration
