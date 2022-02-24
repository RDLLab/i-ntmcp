"""Random policy implementations """
import random
from typing import Dict, Optional, Tuple

import intmcp.model as M

from intmcp.policy import BasePolicy, ActionDist


class RandomPolicy(BasePolicy):
    """Uniform random policy """

    def get_action(self) -> M.Action:
        return random.choice(self.action_space)

    def get_action_init_values(self,
                               history: M.AgentHistory
                               ) -> Dict[M.Action, Tuple[float, int]]:
        return {a: (0.0, 0) for a in self.action_space}

    def get_value(self, history: Optional[M.AgentHistory]) -> float:
        return 0.0

    def get_pi_by_history(self,
                          history: Optional[M.AgentHistory] = None
                          ) -> ActionDist:
        pr_a = 1.0 / len(self.action_space)
        return {a: pr_a for a in self.action_space}
