"""A grid in the Runner Chaser Problem """
from queue import Queue
from typing import Tuple, List, Set, Optional, Union, Dict, Iterable

Loc = int
Dir = int
Coord = Tuple[int, int]

# Direction ENUMS
NORTH = 0
SOUTH = 1
EAST = 2
WEST = 3

DIRS = [NORTH, SOUTH, EAST, WEST]
DIR_STRS = ["N", "S", "E", "W"]

FOV_EXPANSION_INCR = 3


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
    """A grid for the Pursuit Evasion Problem """

    def __init__(self,
                 grid_height: int,
                 grid_width: int,
                 block_locs: Set[Loc],
                 runner_start_locs: List[Loc],
                 runner_goal_locs: Union[List[Loc], Dict[Loc, List[Loc]]],
                 chaser_start_locs: List[Loc]):
        self.height = grid_height
        self.width = grid_width
        self.block_locs = block_locs
        self.runner_start_locs = runner_start_locs
        self._runner_goal_locs = runner_goal_locs
        self.chaser_start_locs = chaser_start_locs

        # set of locations where an agent can visit
        self.valid_locs = set(self.locs)
        self.valid_locs.difference_update(self.block_locs)

    @property
    def locs(self) -> List[Loc]:
        """The list of all locations on grid """
        return list(range(self.num_locs))

    @property
    def num_locs(self) -> int:
        """The number of possible locations on grid, incl. block locs """
        return self.height * self.width

    @property
    def runner_goal_locs(self) -> List[Loc]:
        """The list of all possible runner goal locations """
        if isinstance(self._runner_goal_locs, list):
            return self._runner_goal_locs
        all_locs = set()
        for v in self._runner_goal_locs.values():
            all_locs.update(v)
        return list(all_locs)

    def get_runner_goal_locs(self, runner_start_loc: Loc) -> List[Loc]:
        """The list of all possible runner goal locations on grid conditioned
        on on the runner start location
        """
        if isinstance(self._runner_goal_locs, list):
            return self._runner_goal_locs
        return self._runner_goal_locs[runner_start_loc]

    def get_ascii_repr(self) -> List[List[str]]:
        """Get ascii repr of grid (not including chaser and runner locs) """
        grid_repr = []
        for row in range(self.height):
            row_repr = []
            for col in range(self.width):
                loc = self.coord_to_loc((row, col))
                if loc in self.block_locs:
                    row_repr.append("#")
                else:
                    row_repr.append(".")
            grid_repr.append(row_repr)
        return grid_repr

    def display_init_grid(self) -> None:
        """Display the initial grid, including all possible init locs """
        grid_repr = self.get_ascii_repr()

        loc_lists = [
            self.runner_start_locs,
            self.chaser_start_locs,
            self.runner_goal_locs
        ]
        symbols = ['r', 'c', 'g']
        for locs, symbol in zip(loc_lists, symbols):
            for loc in locs:
                row, col = self.loc_to_coord(loc)
                grid_repr[row][col] = symbol

        output = "\n" + "\n".join(list(list((" ".join(r) for r in grid_repr))))
        print(output)

    def loc_to_coord(self, loc: Loc) -> Coord:
        """Get the (row, col) coords corresponding to location """
        assert 0 <= loc < self.num_locs
        return loc_to_coord(loc, self.width)

    def coord_to_loc(self, coord: Coord) -> Loc:
        """Get the location corresponding to (row, col) coords """
        assert 0 <= coord[0] < self.height
        assert 0 <= coord[1] < self.width
        return coord_to_loc(coord, self.width)

    def manhattan_dist(self, loc1: Loc, loc2: Loc) -> int:
        """Get manhattan distance between two locations on the grid """
        return manhattan_dist(loc1, loc2, self.width)

    def get_neighbouring_locs(self,
                              loc: Loc,
                              include_blocks: bool,
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

        if include_blocks:
            return neighbours

        for i in range(len(neighbours), 0, -1):
            if neighbours[i-1] in self.block_locs:
                neighbours.pop(i-1)

        return neighbours

    def get_neighbour(self,
                      loc: Loc,
                      in_dir: Dir,
                      include_blocks: bool) -> Optional[Loc]:
        """Get the neighbour loc of given loc in given direction """
        assert NORTH <= in_dir <= WEST, f"Invalid direction '{in_dir}'"
        neighbour: Optional[Loc] = None
        if in_dir == NORTH and loc >= self.width:
            neighbour = loc - self.width
        if in_dir == SOUTH and loc < (self.height-1)*self.width:
            neighbour = loc + self.width
        if in_dir == EAST and loc % self.width < self.width-1:
            neighbour = loc + 1
        if in_dir == WEST and loc % self.width > 0:
            neighbour = loc - 1

        if neighbour is None or include_blocks:
            return neighbour

        if neighbour in self.block_locs:
            return None
        return neighbour

    def get_locs_within_dist(self,
                             loc: Loc,
                             dist: int,
                             include_blocks: bool,
                             include_origin: bool) -> Set[Loc]:
        """Get set of locs within given distance from loc """
        assert dist > 0
        adj_locs = self.get_neighbouring_locs(loc, include_blocks)
        in_dist_locs = set(adj_locs)
        if include_origin and (include_blocks or loc not in self.block_locs):
            in_dist_locs.add(loc)

        if dist == 1:
            return in_dist_locs

        for adj_loc in adj_locs:
            in_dist_locs.update(self.get_locs_within_dist(
                adj_loc, dist-1, include_blocks, False
            ))

        if not include_origin and loc in in_dist_locs:
            # must remove since it will get added again during recursive call
            in_dist_locs.remove(loc)

        return in_dist_locs

    def get_min_dist_locs(self,
                          loc: Loc,
                          other_locs: Iterable[Loc]) -> List[Loc]:
        """Get list of closest locs in other_locs to given loc """
        dists: Dict[int, List[Loc]] = {}
        for l2 in other_locs:
            d = self.manhattan_dist(loc, l2)
            if d not in dists:
                dists[d] = []
            dists[d].append(l2)
        if len(dists) == 0:
            return []
        return dists[min(dists)]


def get_fov(ego_loc: Loc, ego_dir: Dir, grid: Grid) -> Set[Loc]:
    """Get the Field of vision from ego loc in ego direction

    Uses BFS starting from ego loc and expanding in the direction the ego
    agent is facing to get the field of vision.
    """
    fov = set([ego_loc])

    ego_coords = grid.loc_to_coord(ego_loc)
    frontier_queue: Queue[Coord] = Queue()
    frontier_queue.put(ego_coords)
    visited = set([ego_coords])

    while not frontier_queue.empty():
        coords = frontier_queue.get()

        for next_coords in _get_fov_successors(
            ego_coords, ego_dir, coords, grid
        ):
            if next_coords not in visited:
                visited.add(next_coords)
                frontier_queue.put(next_coords)
                fov.add(grid.coord_to_loc(next_coords))
    return fov


def _get_fov_successors(ego_coords: Coord,
                        agent_dir: Dir,
                        coords: Coord,
                        grid: Grid) -> List[Coord]:
    successors = []

    forward_successor = _get_fov_successor(coords, agent_dir, grid)
    if forward_successor is not None:
        successors.append(forward_successor)
    else:
        return successors

    if not _check_expand_fov(ego_coords, coords):
        return successors

    side_successor: Optional[Coord] = None
    side_coords_list: List[Coord] = []
    if agent_dir in [NORTH, SOUTH]:
        if 0 < coords[1] <= ego_coords[1]:
            side_coords_list.append((coords[0], coords[1]-1))
        if ego_coords[1] <= coords[1] < grid.width-1:
            side_coords_list.append((coords[0], coords[1]+1))
    if agent_dir in [EAST, WEST]:
        if 0 < coords[0] <= ego_coords[0]:
            side_coords_list.append((coords[0]-1, coords[1]))
        elif ego_coords[0] <= coords[0] < grid.height-1:
            side_coords_list.append((coords[0]+1, coords[1]))

    for side_coord in side_coords_list:
        side_loc = grid.coord_to_loc(side_coord)
        if side_loc in grid.block_locs:
            continue

        side_successor = _get_fov_successor(side_coord, agent_dir, grid)
        if side_successor is not None:
            successors.append(side_successor)

    return successors


def _get_fov_successor(coords: Coord,
                       agent_dir: Dir,
                       grid: Grid) -> Optional[Coord]:
    new_coords = list(coords)
    if agent_dir == NORTH:
        new_coords[0] = coords[0]-1
    elif agent_dir == SOUTH:
        new_coords[0] = coords[0]+1
    elif agent_dir == EAST:
        new_coords[1] = coords[1]+1
    elif agent_dir == WEST:
        new_coords[1] = coords[1]-1

    if (
        not 0 <= new_coords[0] <= grid.height-1
        or not 0 <= new_coords[1] <= grid.width-1
    ):
        return None

    new_loc = grid.coord_to_loc((new_coords[0], new_coords[1]))
    if new_loc in grid.block_locs:
        return None

    return (new_coords[0], new_coords[1])


def _check_expand_fov(ego_coords: Coord, coords: Coord) -> bool:
    # Expands field of vision at depth 1 and
    # then every FOV_EXPANSION_INCR depth
    d = max(abs(ego_coords[0] - coords[0]), abs(ego_coords[1] - coords[1]))
    return d == 1 or (d > 1 and d % FOV_EXPANSION_INCR == 0)


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
    return Grid(
        grid_height=7,
        grid_width=7,
        block_locs=block_locs,
        runner_start_locs=[45],
        runner_goal_locs=[0, 20],
        chaser_start_locs=[4]
    )


def get_pe8_grid() -> Grid:
    """Generate the 8-by-8 PE grid layout.

    This is a discrete version of the map layout from:
    Seaman et al 2018, 'Nested Reasoning About Autonomous Agents Using
    Probabilistic Programs'

    %%%%%%%%%%
    %  9 #8 #%
    %# #    #%
    %# ## # 7%
    %  6   # %
    %# #5# # %
    %  4#3   %
    %#    #2 %
    %0 #1### %
    %%%%%%%%%%

    """
    ascii_map = (
        "  9 #8 #"
        "# #    #"
        "# ## # 7"
        "  6   # "
        "# #5# # "
        "  4#3   "
        "#    #2 "
        "0 #1### "
    )

    # TODO update end locs...
    return _convert_map_to_grid(ascii_map, 8, 8)


def get_pe8v2_grid() -> Grid:
    """Generate the 8-by-8 PE grid layout.

    This is a discrete version of the map layout from:
    Seaman et al 2018, 'Nested Reasoning About Autonomous Agents Using
    Probabilistic Programs'

    %%%%%%%%%%
    %  9 #8 #%
    %# #    #%
    %# ## # 7%
    %  6   # %
    %# #5# # %
    %   #    %
    %#    #2 %
    %0 #1### %
    %%%%%%%%%%

    - 0, 1, 2, 7, 8, 9 are possible runner start and goal locations
    - 5, 6 are possible chaser start locations

    The runner start and goal locations are always on opposite sides of the
    map.
    """
    ascii_map = (
        "  9 #8 #"
        "# #    #"
        "# ## # 7"
        "  6   # "
        "# #5# # "
        "   #    "
        "#    #2 "
        "0 #1### "
    )

    return _convert_map_to_grid(ascii_map, 8, 8)


def get_pe16_grid() -> Grid:
    """Generate the 16-by-16 PE grid layout.

    This is a discrete version of the map layout from:
    Seaman et al 2018, 'Nested Reasoning About Autonomous Agents Using
    Probabilistic Programs'
    """
    ascii_map = (
        "  ## ####### ###"
        "    9##    8 ###"
        "##       # # ###"
        "   ## ##      ##"
        "## ## ##   ## ##"
        "#  #  ##   ## 7#"
        "# ## ###       #"
        "    6       ## #"
        "##  ## ##  ##  #"
        "##   #5##  #   #"
        "##  4  #3    # #"
        "   # #   ##  2 #"
        "#### #   ##   ##"
        "#0    # 1     #"
        "  #### ##  ##   "
        "###     ######  "
    )
    # TODO update start locs ...
    return _convert_map_to_grid(ascii_map, 16, 16)


def get_pe16v2_grid() -> Grid:
    """Generate the 16-by-16 PE grid layout.

    This is a discrete version of the map layout from:
    Seaman et al 2018, 'Nested Reasoning About Autonomous Agents Using
    Probabilistic Programs'

    - 0, 1, 2, 7, 8, 9 are possible runner start and goal locations
    - 5, 6 are possible chaser start locations

    The runner start and goal locations are always on opposite sides of the
    map.

    """
    ascii_map = (
        "  ## ####### ###"
        "    9##    8 ###"
        "##       # # ###"
        "## ## ##      ##"
        "## ## ##   ## ##"
        "#  #  ##   ## 7#"
        "# ## ###       #"
        "    6       ## #"
        "##  ## ##  ##  #"
        "##   #5##  #   #"
        "## #   #     # #"
        "   # #   ##  2 #"
        "## # #   ##   ##"
        "#0     # 1     #"
        "  #### ##  ##   "
        "###     ######  "
    )
    return _convert_map_to_grid(ascii_map, 16, 16)


def get_pe16v3_grid() -> Grid:
    """Generate the 16-by-16 PE grid layout.

    This is a discrete version of the map layout from:
    Seaman et al 2018, 'Nested Reasoning About Autonomous Agents Using
    Probabilistic Programs'

    - 0, 1, 2, 7, 8, 9 are possible runner start and goal locations
    - 5, 6 are possible chaser start locations

    The runner start and goal locations are always on opposite sides of the
    map.

    """
    ascii_map = (
        "  ## ####### ###"
        "    9##    8 ###"
        "##       # # ###"
        "## ## ##      ##"
        "## ## ##   ## ##"
        "#  #  ##   ## 7#"
        "# ## ###   #   #"
        "  # 6        # #"
        "#   ## ##  ##  #"
        "##   #5##  #   #"
        "## #   #     # #"
        "   # #   ##  2 #"
        "## # #   ##   ##"
        "#0     # 1     #"
        "  #### ##  ##   "
        "###     ######  "
    )
    return _convert_map_to_grid(ascii_map, 16, 16)


def get_pe32v2_grid() -> Grid:
    """Generate the 32-by-32 PE grid layout.

    This is a discrete version of the map layout from:
    Seaman et al 2018, 'Nested Reasoning About Autonomous Agents Using
    Probabilistic Programs'

    - 0, 1, 2, 7, 8, 9 are possible runner start and goal locations
    - 5, 6 are possible chaser start locations

    The runner start and goal locations are always on opposite sides of the
    map.

    """
    ascii_map = (
        "       #  ########### #         "
        "   #####  ########### #  #######"
        "      #   ######      8  #######"
        " #       9#               ######"
        " ##              #### ##  ######"
        " ##   #####  ##  #### ##   #####"
        " ##   #####  ##  ##        #####"
        "      ##### ####      ###  #####"
        " ### ###### ####      ###   ####"
        " ### #####  ####      ####  ####"
        " ##  ##### #####      ##### 7###"
        " ##  ####  #####       ####  ###"
        " #  #####  #####       ####  ###"
        " #  ##### ######       #     ###"
        " #       6               ##  ###"
        "    #       #   ##      ####  ##"
        "     #   ####  ###     ####   ##"
        "#### #   ####  ###    ####    ##"
        "#### #   ### 5####   ####     ##"
        "####  #      ####     ##  #   ##"
        "##### #      ####        ##   ##"
        "#                          ##  #"
        "         ###        ##    2 #  #"
        "  ###### ###      #### ## #    #"
        "########  ##      ####    #  #  "
        "########  ##       ####     ####"
        "###          ##   1##        ###"
        "          #  ####       #       "
        "   0  ###### ######   ####      "
        "  ########## #        #####     "
        "##########      ############    "
        "#                               "
    )
    return _convert_map_to_grid(ascii_map, 32, 32)


def _convert_map_to_grid(ascii_map: str,
                         height: int,
                         width: int,
                         block_symbol: str = "#",
                         chaser_start_symbols: Optional[Set[str]] = None,
                         runner_end_symbols: Optional[Set[str]] = None,
                         runner_goal_symbol_map: Optional[Dict] = None
                         ) -> Grid:
    assert len(ascii_map) == height * width

    if chaser_start_symbols is None:
        chaser_start_symbols = set(['3', '4', '5', '6'])
    if runner_end_symbols is None:
        runner_end_symbols = set(['0', '1', '2', '7', '8', '9'])
    if runner_goal_symbol_map is None:
        runner_goal_symbol_map = {
            '0': ['7', '8', '9'],
            '1': ['7', '8', '9'],
            '2': ['8', '9'],
            '7': ['0', '1'],
            '8': ['0', '1', '2'],
            '9': ['0', '1', '2'],
        }

    block_locs = set()
    runner_end_locs = []
    chaser_start_locs = []

    runner_symbol_loc_map = {}

    for loc, symbol in enumerate(ascii_map):
        if symbol == block_symbol:
            block_locs.add(loc)
        elif symbol in chaser_start_symbols:
            chaser_start_locs.append(loc)
        elif symbol in runner_end_symbols:
            runner_end_locs.append(loc)
            runner_symbol_loc_map[symbol] = loc

    runner_goal_locs_map = {}
    for start_symbol, goal_symbols in runner_goal_symbol_map.items():
        start_loc = runner_symbol_loc_map[start_symbol]
        runner_goal_locs_map[start_loc] = [
            runner_symbol_loc_map[symbol] for symbol in goal_symbols
        ]

    return Grid(
        grid_height=height,
        grid_width=width,
        block_locs=block_locs,
        runner_start_locs=runner_end_locs,
        runner_goal_locs=runner_goal_locs_map,
        chaser_start_locs=chaser_start_locs
    )


SUPPORTED_GRIDS = {
    'cap7': get_cap7_grid,
    '8by8': get_pe8_grid,
    '8by8v2': get_pe8v2_grid,
    '16by16': get_pe16_grid,
    '16by16v2': get_pe16v2_grid,
    '16by16v3': get_pe16v3_grid,
    '32by32v2': get_pe32v2_grid
}


def load_grid(grid_name: str) -> 'Grid':
    """Load grid with given name """
    grid_name = grid_name.lower()
    assert grid_name in SUPPORTED_GRIDS, (
        f"Unsupported grid name '{grid_name}'. Grid name must be one of: "
        f"{SUPPORTED_GRIDS.keys()}."
    )
    return SUPPORTED_GRIDS[grid_name]()
