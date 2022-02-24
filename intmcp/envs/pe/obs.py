"""Observations for the Discrete Persuit Evasion problem

The field of vision for agents looks like this:

       ***
    ****
  ******
>*******
  ******
    ****
       *

I.e. the width increases by 1 on each side for when distance d:
- d < 3 and d == 1
- d >= 3 and d % 3 == 0

With objects blocking the view

       **
  # ****#
  ****#
>****#
  **#

"""
from typing import Tuple

from intmcp.model import Observation

from intmcp.envs.pe import grid as grid_lib

WallObs = Tuple[bool, bool, bool, bool]


class PEChaserObs(Observation):
    """A chaser observation for the Discrete Pursuit Evasion problem

    Includes an observation of:
    - whether or not the runner detected in field of vision
    - whether or not runner has been heard (i.e. runner is within Manhattan
      distance of 2)
    - whether there is a wall in each of the cardinal directions
    """

    def __init__(self, walls: WallObs, seen: bool, heard: bool):
        self.walls = walls
        self.seen = seen
        self.heard = heard

    def __str__(self):
        return f"<{self.walls}, {self.seen}, {self.heard}>"

    def __eq__(self, other):
        return (
            (self.walls, self.seen, self.heard)
            == (other.walls, other.seen, other.heard)
        )

    def __hash__(self):
        return hash((self.walls, self.seen, self.heard))


class PERunnerObs(Observation):
    """A Runner observation for the Discrete Pursuit Evasion problem

    Includes an observation of:
    - runners goal location
    - whether or not the chaser detected in field of vision
    - whether or not chaser has been heard (i.e. runner is within Manhattan
      distance of 2)
    - whether there is a wall in each of the cardinal directions
    """

    def __init__(self,
                 walls: WallObs,
                 seen: bool,
                 heard: bool,
                 goal: grid_lib.Loc):
        self.walls = walls
        self.seen = seen
        self.heard = heard
        self.goal = goal

    def __str__(self):
        return f"<{self.walls}, {self.seen}, {self.heard}, {self.goal}>"

    def __eq__(self, other):
        return (
            (self.walls, self.seen, self.heard, self.goal)
            == (other.walls, other.seen, other.heard, other.goal)
        )

    def __hash__(self):
        return hash((self.walls, self.seen, self.heard, self.goal))
