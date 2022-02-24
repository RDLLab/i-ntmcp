"""Observations for the Runner Chaser problem """
from typing import Tuple

from intmcp.model import Observation

from intmcp.envs.rc import grid as grid_lib

# Cell obs
OPPONENT = 0
WALL = 1
EMPTY = 2

CELL_OBS = [OPPONENT, WALL, EMPTY]
CELL_OBS_STR = ["X", "#", "0"]


class RCObs(Observation):
    """An observation for the Runner Chaser problem

    Includes an observation of the agents start location and the contents of
    the adjacent cells.
    """

    def __init__(self,
                 loc: grid_lib.Loc,
                 adj_obs: Tuple[int, ...]):
        self.loc = loc
        self.adj_obs = adj_obs

    def __str__(self):
        adj_obs_str = ",".join(CELL_OBS_STR[i] for i in self.adj_obs)
        return f"<{self.loc},({adj_obs_str})>"

    def __eq__(self, other):
        return (
            (self.loc, self.adj_obs)
            == (other.loc, other.adj_obs)
        )

    def __hash__(self):
        return hash((self.loc, self.adj_obs))
