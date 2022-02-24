"""The model data structure """
import abc
from argparse import ArgumentParser
from typing import Optional, List, Tuple, Callable

from intmcp.model.state import State
from intmcp.model.belief import Belief
from intmcp.model.parts import AgentID, Outcomes
from intmcp.model.action import JointAction, Action
from intmcp.model.observation import JointObservation, Observation

JointTimestep = Tuple[State, JointObservation, List[float], bool]
ReinvigorateFunction = Callable[
    [AgentID, State, JointAction, Observation], Tuple[State, JointObservation]
]


class POSGModel(abc.ABC):
    """A Partially Observable Stochastic Game model """

    # pylint: disable=unused-argument
    def __init__(self, num_agents: int, **kwargs):
        self.num_agents = num_agents
        # generate these as requested as they may be very large for some envs
        self.agent_action_spaces: List[List[Action]] = []
        self.agent_obs_spaces: List[List[Observation]] = []

    @property
    @abc.abstractmethod
    def state_space(self) -> List[State]:
        """The POSG state space """

    @property
    @abc.abstractmethod
    def obs_space(self) -> List[JointObservation]:
        """The POSG joint observation space """

    @property
    @abc.abstractmethod
    def action_space(self) -> List[JointAction]:
        """The POSG joint action space """

    @property
    @abc.abstractmethod
    def initial_belief(self) -> Belief:
        """The POMDP initial belief """

    @property
    @abc.abstractmethod
    def r_min(self) -> float:
        """The Minimum possible reward value min_s,a R(s, a) """

    @property
    @abc.abstractmethod
    def r_max(self) -> float:
        """The Maximumm possible reward value max_s,a R(s, a) """

    @property
    def reinvigorate_fn(self) -> Optional[ReinvigorateFunction]:
        """The model reinvigorate function or None if model has no function

        The returned function reinvigorates a state to produce a new state
        consistent with given agent's action and observation.

        Will raise IndexError if no valid consistent state can be found.
        """
        return None

    def reset(self) -> Tuple[State, JointObservation]:
        """Perform any reset action for the model between episodes.

        Returns an sampled initial state and initial joint observation.

        By default this function does nothing (in terms of reseting model
        state) and calls the sample_initial_state() and sample_initial_obs()
        functions in order.

        It should be overwritten if additional reset functionallity is
        required.
        """
        state = self.sample_initial_state()
        joint_obs = self.sample_initial_obs(state)
        return state, joint_obs

    @abc.abstractmethod
    def get_init_belief(self,
                        agent_id: AgentID,
                        obs: Observation) -> Belief:
        """Get the initial obs conditioned belief for a given agent """

    @abc.abstractmethod
    def sample_initial_state(self) -> State:
        """Sample an initial state from initial belief """

    @abc.abstractmethod
    def sample_initial_obs(self, state: State) -> JointObservation:
        """Sample an initial observation given initial state """

    @abc.abstractmethod
    def step(self, state: State, action: JointAction) -> JointTimestep:
        """Perform generative step """

    @abc.abstractmethod
    def is_terminal(self, state: State) -> bool:
        """Check if state is a terminal state """

    @abc.abstractmethod
    def get_outcome(self, state: State) -> List[Outcomes]:
        """Get outcome for each agent for given state.

        This function can be used to specify if episode ended in win/draw/loss,
        etc.

        By convention a value of:
        - 1 = win/success
        - 0 = draw
        - -1 = loss/failure
        - None = undefined
        """

    def get_agent_action_space(self, agent_id: AgentID) -> List[Action]:
        """Get the action space for an agent """
        if len(self.agent_action_spaces) == 0:
            for i in range(self.num_agents):
                self.agent_action_spaces.append(
                    JointAction.get_agent_action_space(self.action_space, i)
                )
        return self.agent_action_spaces[agent_id]

    def get_agent_obs_space(self, agent_id: AgentID) -> List[Observation]:
        """Get the action space for an agent """
        if len(self.agent_obs_spaces) == 0:
            for i in range(self.num_agents):
                self.agent_obs_spaces.append(
                    JointObservation.get_agent_obs_space(self.obs_space, i)
                )
        return self.agent_obs_spaces[agent_id]

    @classmethod
    def get_args_parser(cls,
                        parser: Optional[ArgumentParser] = None
                        ) -> ArgumentParser:
        """Get or update argument parser for creating the model. """
        if parser is None:
            parser = ArgumentParser()
        return parser
