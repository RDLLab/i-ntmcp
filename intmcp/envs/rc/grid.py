"""A grid in the Runner Chaser Problem """
from typing import Tuple, List, Set, Optional

Loc = int
Coord = Tuple[int, int]

# Direction ENUMS
NORTH = 0
SOUTH = 1
EAST = 2
WEST = 3

DIRS = [NORTH, SOUTH, EAST, WEST]
DIR_STRS = ["N", "S", "E", "W"]


def loc_to_coord(loc: Loc, grid_width: int) -> Coord:
    """Get the (row, col) coords corresponding to location """
    return (loc // grid_width, loc % grid_width)


def coord_to_loc(coord: Coord, grid_width: int) -> Loc:
    """Get the location corresponding to (row, col) coords """
    return coord[0]*grid_width + coord[1]


def manhattan_dist(loc1: Loc, loc2: Loc, grid_width: int) -> int:
    """Get manhattan distance between two locations on the grid """
    return (
        abs(loc1 // grid_width - loc2 // grid_width)
        + abs(loc1 % grid_width - loc2 % grid_width)
    )


class Grid:
    """A grid for the Runner Chaser Problem """

    def __init__(self,
                 grid_height: int,
                 grid_width: int,
                 block_locs: Set[Loc],
                 safe_locs: Set[Loc],
                 init_runner_locs: List[Loc],
                 init_chaser_locs: List[Loc],
                 valid_chaser_locs: Optional[Set[Loc]] = None,
                 valid_runner_locs: Optional[Set[Loc]] = None):
        self.height = grid_height
        self.width = grid_width
        self.size = grid_height * grid_width
        self.block_locs = block_locs
        self.safe_locs = safe_locs
        self.init_runner_locs = init_runner_locs
        self.init_chaser_locs = init_chaser_locs

        if valid_chaser_locs is None:
            valid_chaser_locs = set(self.locs)
            valid_chaser_locs.difference_update(self.block_locs)
            valid_chaser_locs.difference_update(self.safe_locs)
        self.valid_chaser_locs = valid_chaser_locs

        if valid_runner_locs is None:
            valid_runner_locs = set(self.locs)
            valid_runner_locs.difference_update(self.block_locs)
            valid_runner_locs.difference_update(self.safe_locs)
        self.valid_runner_locs = valid_runner_locs

    @property
    def locs(self) -> List[Loc]:
        """The list of all locations on grid """
        return list(range(self.height*self.width))

    def get_all_unblocked_locs(self) -> List[Loc]:
        """Get the list of all locations on the grid excluding blocks """
        return [loc for loc in self.locs if loc not in self.block_locs]

    def get_ascii_repr(self) -> List[List[str]]:
        """Get ascii repr of grid (not including chaser and runner locs) """
        grid_repr = []
        for row in range(self.height):
            row_repr = []
            for col in range(self.width):
                loc = self.coord_to_loc((row, col))
                if loc in self.safe_locs:
                    row_repr.append("S")
                elif loc in self.block_locs:
                    row_repr.append("#")
                else:
                    row_repr.append(".")
            grid_repr.append(row_repr)
        return grid_repr

    def display_init_grid(self) -> None:
        """Display the initial grid, including all possible init locs """
        grid_repr = self.get_ascii_repr()

        for loc in self.init_chaser_locs:
            chaser_row, chaser_col = self.loc_to_coord(loc)
            grid_repr[chaser_row][chaser_col] = "C"

        for loc in self.init_runner_locs:
            runner_row, runner_col = self.loc_to_coord(loc)
            grid_repr[runner_row][runner_col] = "R"

        output = "\n" + "\n".join(list(list((" ".join(r) for r in grid_repr))))
        print(output)

    def loc_to_coord(self, loc: Loc) -> Coord:
        """Get the (row, col) coords corresponding to location """
        return loc_to_coord(loc, self.width)

    def coord_to_loc(self, coord: Coord) -> Loc:
        """Get the location corresponding to (row, col) coords """
        return coord_to_loc(coord, self.width)

    def manhattan_dist(self, loc1: Loc, loc2: Loc) -> int:
        """Get manhattan distance between two locations on the grid """
        return manhattan_dist(loc1, loc2, self.width)

    def get_neighbouring_locs(self,
                              loc: Loc,
                              ignore_blocks: bool,
                              include_out_of_bounds: bool = False
                              ) -> List[Loc]:
        """Get set of adjacent non-blocked locations """
        neighbours = []
        if loc >= self.width or include_out_of_bounds:
            neighbours.append(loc-self.width)    # N
        if loc < (self.height - 1)*self.width or include_out_of_bounds:
            neighbours.append(loc+self.width)    # S
        if loc % self.width < self.width - 1 or include_out_of_bounds:
            neighbours.append(loc+1)             # E
        if loc % self.width > 0 or include_out_of_bounds:
            neighbours.append(loc-1)             # W

        if ignore_blocks:
            return neighbours

        for i in range(len(neighbours), 0, -1):
            if neighbours[i-1] in self.block_locs:
                neighbours.pop(i-1)

        return neighbours

def get_cap3_grid() -> Grid:
    """Generate the Choose a Path 3-by-3 grid layout.

    3x3
    #####
    #SC #
    # #S#
    # R #
    #####
    """
    block_locs = set([
        4
    ])

    # exclude locs on other side of safe locs
    valid_runner_locs = set([
        # None on row 0
        3,
        6, 7, 8
    ])

    return Grid(
        grid_height=3,
        grid_width=3,
        block_locs=block_locs,
        safe_locs=set([0, 5]),
        init_runner_locs=[7],
        init_chaser_locs=[1],
        valid_chaser_locs=None,
        valid_runner_locs=valid_runner_locs
    )


def get_cap4_grid() -> Grid:
    """Generate the Choose a Path 4-by-4 grid layout.

    4x4
    ######
    #S C #
    # ##S#
    # #  #
    #  R##
    ######
    """
    block_locs = set([
        #
        5, 6,
        9,
        15
    ])

    # exclude locs on other side of safe locs
    valid_runner_locs = set([
        # None on row 0
        4,
        8, 10, 11,
        12, 13, 14,
    ])

    return Grid(
        grid_height=4,
        grid_width=4,
        block_locs=block_locs,
        safe_locs=set([0, 7]),
        init_runner_locs=[14],
        init_chaser_locs=[2],
        valid_chaser_locs=None,
        valid_runner_locs=valid_runner_locs
    )


def get_cap7_grid() -> Grid:
    """Generate the Choose a Path 7-by-7 grid layout.

    7x7
    #########
    #S   C  #
    # ##### #
    # #### S#
    #  ### ##
    ## ### ##
    ##  #  ##
    ### R ###
    #########

    """
    block_locs = set([
        8, 9, 10, 11, 12,
        15, 16, 17, 18,
        23, 24, 25, 27,
        28, 30, 31, 32, 34,
        35, 38, 41,
        42, 43, 47, 48
    ])

    # exclude locs on other side of safe locs
    valid_runner_locs = set([
        # None on row 0
        7,
        14, 19,
        21, 22, 26,
        29, 33,
        36, 37, 39, 40,
        44, 45, 46

    ])

    return Grid(
        grid_height=7,
        grid_width=7,
        block_locs=block_locs,
        safe_locs=set([0, 20]),
        init_runner_locs=[45],
        init_chaser_locs=[4],
        valid_chaser_locs=None,
        valid_runner_locs=valid_runner_locs
    )


SUPPORTED_GRIDS = {
    'cap3': get_cap3_grid,
    'cap4': get_cap4_grid,
    'cap7': get_cap7_grid,
}


def load_grid(grid_name: str) -> 'Grid':
    """Load grid with given name """
    grid_name = grid_name.lower()
    assert grid_name in SUPPORTED_GRIDS, (
        f"Unsupported grid name '{grid_name}'. Grid name must be one of: "
        f"{SUPPORTED_GRIDS.keys()}."
    )
    return SUPPORTED_GRIDS[grid_name]()
