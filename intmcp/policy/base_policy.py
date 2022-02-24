"""Base abstract policy classes that should be implemented by all concrete
policies
"""
import abc
import logging
from argparse import ArgumentParser
from typing import Optional, Dict, Mapping, Any, Tuple

import intmcp.model as M
import intmcp.log as log_lib

StateDist = Dict[M.State, float]
ActionDist = Dict[M.Action, float]


class BasePolicy(abc.ABC):
    """Base policy interface """

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: int,
                 gamma: float,
                 logger: Optional[logging.Logger] = None,
                 **kwargs):
        self.model = model
        self.ego_agent = ego_agent
        self.action_space = model.get_agent_action_space(ego_agent)
        self.gamma = gamma
        self._logger = logging.getLogger() if logger is None else logger
        self.history = M.AgentHistory.get_init_history()
        self.kwargs = kwargs
        self._last_action = M.Action.get_null_action()
        self._statistics: Dict[str, Any] = {}

    def step(self, obs: M.Observation) -> M.Action:
        """Execute a single policy step

        This involves:
        1. a updating policy with last action and given observation
        2. next action using updated policy
        """
        self.update(self._last_action, obs)
        self._last_action = self.get_action()
        return self._last_action

    @abc.abstractmethod
    def get_action(self) -> M.Action:
        """Get action for given obs """

    @abc.abstractmethod
    def get_action_init_values(self,
                               history: M.AgentHistory
                               ) -> Dict[M.Action, Tuple[float, int]]:
        """Get initial visit count and values for each action for a history """

    @abc.abstractmethod
    def get_value(self, history: Optional[M.AgentHistory]) -> float:
        """Get a value estimate of a history """

    def get_action_by_history(self, history: M.AgentHistory) -> M.Action:
        """Get action given history, leaving state of policy unchanged """
        current_history = self.history
        self.reset_history(history)
        action = self.get_action()
        self.reset_history(current_history)
        return action

    def get_pi_by_history(self,
                          history: Optional[M.AgentHistory] = None
                          ) -> ActionDist:
        """Get agent's distribution over actions for a given history.

        If history is None or not given then uses current history.

        By default this returns prob 1.0 for the action returned by calling
        the get_action_by_history(), and returns 0.0 for all other actions.
        But can and should be overwritten in subclasses.
        """
        if history is None:
            history = self.history

        action = self.get_action_by_history(history)
        pi = {}
        for a in self.action_space:
            pi[a] = 1.0 if a == action else 0.0
        return pi

    def update(self, action: M.Action, obs: M.Observation) -> None:
        """Update policy history. Should be called before """
        self.history = self.history.extend(action, obs)

    def reset(self) -> None:
        """Reset the policy """
        self.history = M.AgentHistory.get_init_history()

    def reset_history(self, history: M.AgentHistory) -> None:
        """Reset policy history to given history """
        self.history = history

    @classmethod
    def get_args_parser(cls,
                        parser: Optional[ArgumentParser] = None
                        ) -> ArgumentParser:
        """Get or update argument parser for creating the model. """
        if parser is None:
            parser = ArgumentParser()
        return parser

    @classmethod
    def initialize(cls,
                   model: M.POSGModel,
                   ego_agent: int,
                   gamma: float,
                   **kwargs) -> 'BasePolicy':
        """Initialize the policy """
        return cls(model, ego_agent, gamma, **kwargs)

    #######################################################
    # Logging
    #######################################################

    @property
    def statistics(self) -> Mapping[str, Any]:
        """Returns current agent statistics as a dictionary."""
        return self._statistics

    def _log_info1(self, msg: str):
        """Log an info message """
        self._logger.log(log_lib.INFO1, self._format_msg(msg))

    def _log_info2(self, msg: str):
        """Log an info message """
        self._logger.log(log_lib.INFO2, self._format_msg(msg))

    def _log_debug(self, msg: str):
        """Log a debug message """
        self._logger.debug(self._format_msg(msg))

    def _log_debug1(self, msg: str):
        """Log a debug2 message """
        self._logger.log(log_lib.DEBUG1, self._format_msg(msg))

    def _log_debug2(self, msg: str):
        """Log a debug2 message """
        self._logger.log(log_lib.DEBUG2, self._format_msg(msg))

    def _format_msg(self, msg: str):
        return f"i={self.ego_agent} {msg}"

    def __str__(self):
        return self.__class__.__name__
