"""State in the Discrete Pursuit Evasion Problem """
from intmcp.model import State

from intmcp.envs.pe import grid as grid_lib


class PEState(State):
    """A state in the Discrete Pursuit Evasion Problem """

    def __init__(self,
                 runner_loc: grid_lib.Loc,
                 chaser_loc: grid_lib.Loc,
                 runner_dir: grid_lib.Dir,
                 chaser_dir: grid_lib.Dir,
                 runner_goal_loc: grid_lib.Loc,
                 grid: grid_lib.Grid):
        super().__init__()
        self.runner_loc = runner_loc
        self.chaser_loc = chaser_loc
        self.runner_dir = runner_dir
        self.chaser_dir = chaser_dir
        self.runner_goal_loc = runner_goal_loc
        self.grid = grid

    def copy(self) -> 'PEState':
        """Get a copy of this state """
        return PEState(
            self.runner_loc,
            self.chaser_loc,
            self.runner_dir,
            self.chaser_dir,
            self.runner_goal_loc,
            self.grid
        )

    def render_asci(self):
        grid_repr = self.grid.get_ascii_repr()
        c_fov = grid_lib.get_fov(self.chaser_loc, self.chaser_dir, self.grid)
        r_fov = grid_lib.get_fov(self.runner_loc, self.runner_dir, self.grid)

        for fov_set, symbol in zip(
                [
                    c_fov.difference(r_fov),
                    r_fov.difference(c_fov),
                    c_fov.intersection(r_fov)
                ],
                ["^", "*", 'o']
        ):
            for loc in fov_set:
                coord = self.grid.loc_to_coord(loc)
                grid_repr[coord[0]][coord[1]] = symbol

        runner_coords = self.grid.loc_to_coord(self.runner_loc)
        chaser_coords = self.grid.loc_to_coord(self.chaser_loc)
        runner_goal_coords = self.grid.loc_to_coord(self.runner_goal_loc)

        grid_repr[runner_goal_coords[0]][runner_goal_coords[1]] = "G"
        grid_repr[chaser_coords[0]][chaser_coords[1]] = "C"

        if self.runner_loc == self.chaser_loc or self.runner_loc in c_fov:
            grid_repr[runner_coords[0]][runner_coords[1]] = "X"
        else:
            grid_repr[runner_coords[0]][runner_coords[1]] = "R"

        return (
            str(self)
            + "\n" + "\n".join(list(list((" ".join(r) for r in grid_repr))))
        )

    def __str__(self):
        return (
            f"<{self.runner_loc}, {self.chaser_loc}, "
            f"{grid_lib.DIR_STRS[self.runner_dir]}, "
            f"{grid_lib.DIR_STRS[self.chaser_dir]}, "
            f"{self.runner_goal_loc}>"
        )

    def __hash__(self):
        return hash((
            self.runner_loc,
            self.chaser_loc,
            self.runner_dir,
            self.chaser_dir,
            self.runner_goal_loc
        ))

    def __eq__(self, other):
        return (
            self.runner_loc == other.runner_loc
            and self.chaser_loc == other.chaser_loc
            and self.runner_dir == other.runner_dir
            and self.chaser_dir == other.chaser_dir
            and self.runner_goal_loc == other.runner_goal_loc
        )
