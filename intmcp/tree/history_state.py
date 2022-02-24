"""A History State

Each History state is made up of the state of the environment and
the joint history of the environment that lead to that state
"""

import intmcp.model as M


class HistoryState(M.State):
    """A History State

    A history state is made up of the state of the environment and
    the joint history of the environment that lead to that state
    """

    def __init__(self,
                 state: M.State,
                 history: M.JointHistory):
        super().__init__()
        self.state = state
        self.history = history

    def __hash__(self):
        return hash((self.state, self.history))

    def __eq__(self, other):
        if not isinstance(other, HistoryState):
            return False
        return self.state == other.state and self.history == other.history

    def __str__(self):
        return f"[s={self.state}, h={self.history}]"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def initial_belief(cls, initial_posg_belief: M.Belief, num_agents: int):
        """Generate Initial Iterative Belief from POSG Belief.

        The initial iterative state belief includes the initial joint history
        along with the initial state
        """
        def initial_belief_fn():
            state = initial_posg_belief.sample()
            history = M.JointHistory.get_init_history(num_agents)
            return cls(state, history)

        return M.InitialParticleBelief(initial_belief_fn)
