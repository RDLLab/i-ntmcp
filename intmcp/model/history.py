"""A history in the environment """
from typing import Tuple, Optional

from intmcp.model import Action, Observation, JointAction, JointObservation


class AgentHistory:
    """An Action-Observation history for a single agent in a POSG environment.

    A History is an ordered Tuple of (Action, Observation) tuples with one
    entry for each time step in the environment
    """

    def __init__(self, history: Tuple[Tuple[Action, Observation], ...]):
        self.history = history
        # pylint: disable=invalid-name
        self.t = len(history) - 1

    def extend(self, action: Action, obs: Observation) -> "AgentHistory":
        """Extend the current history with given action, observation pair. """
        new_history = list(self.history)
        new_history.append((action, obs))
        return AgentHistory(tuple(new_history))

    def get_sub_history(self, horizon: int) -> "AgentHistory":
        """Get a subset of history up to given horizon """
        assert 0 < horizon <= len(self.history), (
            "Cannot get sub history horizon must be 0 < horizon <= "
            f"len(history: 0 < {horizon} <= len(self.history) invalid"
        )
        if horizon == len(self.history):
            return self
        return AgentHistory(self.history[:horizon])

    def get_last_step(self) -> Tuple[Action, Observation]:
        """Get the last step in the history """
        return self.history[-1]

    @classmethod
    def get_init_history(cls,
                         obs: Optional[Observation] = None) -> "AgentHistory":
        """Get Initial history """
        if obs is None:
            return cls(())
        return cls(((Action.get_null_action(), obs), ))

    def __hash__(self):
        return hash(self.history)

    def __eq__(self, other):
        if not isinstance(other, AgentHistory):
            return False
        return self.history == other.history

    def __str__(self):
        h_str = []
        for (a, o) in self.history:
            h_str.append(f"({a}, {o})")
        h_str = ",".join(h_str)
        return f"h_{self.t}=<{h_str}>"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, key):
        return self.history[key]

    def __iter__(self):
        return AgentHistoryIterator(self)


class AgentHistoryIterator:
    """Iterator for the History class """

    def __init__(self, history: AgentHistory):
        self.history = history
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx < len(self.history.history):
            self._idx += 1
            return self.history[self._idx-1]
        raise StopIteration


class JointHistory:
    """A joint history for all agents in the environment """

    def __init__(self, agent_histories: Tuple[AgentHistory, ...]):
        self.agent_histories = agent_histories
        self.num_agents = len(self.agent_histories)

    @classmethod
    def get_init_history(cls,
                         num_agents: int,
                         obs: Optional[JointObservation] = None
                         ) -> "JointHistory":
        """Get Initial joint history """
        if obs is None:
            return cls(tuple(
                AgentHistory.get_init_history() for _ in range(num_agents)
            ))
        return cls(tuple(
            AgentHistory.get_init_history(obs.get_agent_obs(i))
            for i in range(num_agents)
        ))

    def get_agent_history(self, agent_id: int) -> AgentHistory:
        """Get the history of given agent. """
        return self.agent_histories[agent_id]

    def extend(self,
               action: JointAction,
               obs: JointObservation) -> "JointHistory":
        """Extend the current history with given action, observation pair. """
        new_agent_histories = []
        for i in range(self.num_agents):
            new_agent_histories.append(self.agent_histories[i].extend(
                action.get_agent_action(i), obs.get_agent_obs(i)
            ))
        return JointHistory(tuple(new_agent_histories))

    def get_sub_history(self, horizon: int) -> "JointHistory":
        """Get a subset of history up to given horizon """
        sub_agent_histories = []
        for i in range(self.num_agents):
            sub_agent_histories.append(
                self.agent_histories[i].get_sub_history(horizon)
            )
        return JointHistory(tuple(sub_agent_histories))

    def __hash__(self):
        return hash(self.agent_histories)

    def __eq__(self, other):
        if not isinstance(other, JointHistory):
            return False
        return self.agent_histories == other.agent_histories

    def __str__(self):
        h_str = [f"{i} {h}" for i, h in enumerate(self.agent_histories)]
        h_str.insert(0, "<JointHistory:")
        h_str.append(">")
        return "\n".join(h_str)

    def __repr__(self):
        return self.__str__()
