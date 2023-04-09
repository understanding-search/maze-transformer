import random
from dataclasses import dataclass
from typing import NamedTuple, cast
import warnings

import numpy as np
from jaxtyping import Float, Int, Bool, Shaped
from muutils.misc import list_split
from muutils.json_serialize.serializable_dataclass import SerializableDataclass, serializable_dataclass, serializable_field
from muutils.tensor_utils import NDArray

from maze_transformer.generation.constants import (
    NEIGHBORS_MASK,
    SPECIAL_TOKENS,
    Coord,
    CoordArray,
    CoordTup,
)
from maze_transformer.utils.token_utils import (
    decode_maze_tokens_to_coords,
    get_adj_list_tokens,
    get_path_tokens,
)

RGB = tuple[int, int, int]

def coord_str_to_tuple(coord_str: str) -> CoordTup:
    """convert a coordinate string to a tuple"""

    stripped: str = coord_str.lstrip("(").rstrip(")")
    return tuple(int(x) for x in stripped.split(","))


@serializable_dataclass(frozen=True, kw_only=True)
class LatticeMaze(SerializableDataclass):
    """lattice maze (nodes on a lattice, connections only to neighboring nodes)

    Connection List represents which nodes (N) are connected in each direction.

    First and second elements represent rightward and downward connections,
    respectively.

    Example:
      Connection list:
        [
          [ # down
            [F T],
            [F F]
          ],
          [ # right
            [T F],
            [T F]
          ]
        ]

      Nodes with connections
        N T N F
        F   T
        N T N F
        F   F

      Graph:
        N - N
            |
        N - N

    Note: the bottom row connections going down, and the
    right-hand connections going right, will always be False.
    """
    connection_list: Bool[np.ndarray, "lattice_dim x y"]
    generation_meta: dict | None = serializable_field(default=None, compare=False)
    lattice_dim: int = serializable_field(default=2)

    grid_shape = property(lambda self: self.connection_list.shape[1:])

    n_connections = property(lambda self: self.connection_list.sum())


    # ============================================================
    # basic methods
    # ============================================================
    @staticmethod
    def heuristic(a: CoordTup, b: CoordTup) -> float:
        """return manhattan distance between two points"""
        return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])

    def nodes_connected(self, a: Coord, b: Coord, /) -> bool:
        """returns whether two nodes are connected"""
        delta: Coord = b - a
        if np.abs(delta).sum() != 1:
            # return false if not even adjacent
            return False
        else:
            # test for wall
            dim: int = np.argmax(np.abs(delta))
            clist_node: Coord = a if (delta.sum() > 0) else b
            return self.connection_list[dim, clist_node[0], clist_node[1]]

    def get_coord_neighbors(self, c: Coord) -> CoordArray:
        neighbors: list[Coord] = [
            neighbor
            for neighbor in (c + NEIGHBORS_MASK)
            if (
                (0 <= neighbor[0] < self.grid_shape[0])  # in x bounds
                and (0 <= neighbor[1] < self.grid_shape[1])  # in y bounds
                and self.nodes_connected(c, neighbor)  # connected
            )
        ]

        return np.array(neighbors)

    def find_shortest_path(
        self,
        c_start: CoordTup,
        c_end: CoordTup,
    ) -> list[Coord]:
        """find the shortest path between two coordinates, using A*"""

        g_score: dict[
            CoordTup, float
        ] = dict()  # cost of cheapest path to node from start currently known
        f_score: dict[CoordTup, float] = {
            c_start: 0.0
        }  # estimated total cost of path thru a node: f_score[c] := g_score[c] + heuristic(c, c_end)

        # init
        g_score[c_start] = 0.0
        g_score[c_start] = self.heuristic(c_start, c_end)

        closed_vtx: set[CoordTup] = set()  # nodes already evaluated
        open_vtx: set[CoordTup] = set([c_start])  # nodes to be evaluated
        source: dict[
            CoordTup, CoordTup
        ] = (
            dict()
        )  # node immediately preceding each node in the path (currently known shortest path)

        while open_vtx:
            # get lowest f_score node
            c_current: CoordTup = min(open_vtx, key=lambda c: f_score[c])
            # f_current: float = f_score[c_current]

            # check if goal is reached
            if c_end == c_current:
                path: list[CoordTup] = [c_current]
                p_current: CoordTup = c_current
                while p_current in source:
                    p_current = source[p_current]
                    path.append(p_current)
                return path[::-1]

            # close current node
            closed_vtx.add(c_current)
            open_vtx.remove(c_current)

            # update g_score of neighbors
            _np_neighbor: Coord
            for _np_neighbor in self.get_coord_neighbors(c_current):
                neighbor: CoordTup = tuple(_np_neighbor)

                if neighbor in closed_vtx:
                    # already checked
                    continue
                g_temp: float = g_score[c_current] + 1  # always 1 for maze neighbors

                if neighbor not in open_vtx:
                    # found new vtx, so add
                    open_vtx.add(neighbor)

                elif g_temp >= g_score[neighbor]:
                    # if already knew about this one, but current g_score is worse, skip
                    continue

                # store g_score and source
                source[neighbor] = c_current
                g_score[neighbor] = g_temp
                f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, c_end)

    def get_nodes(self) -> list[Coord]:
        """return a list of all nodes in the maze"""

        return [
            (row, col)
            for row in range(self.grid_shape[0])
            for col in range(self.grid_shape[1])
        ]

    def generate_random_path(self) -> list[Coord]:
        """ "return a path between randomly chosen start and end nodes"""

        # we can't create a "path" in a single-node maze
        assert self.grid_shape[0] > 1 and self.grid_shape[1] > 1

        start, end = random.sample(self.get_nodes(), 2)
        return self.find_shortest_path(start, end)

    # ============================================================
    # to and from adjacency list
    # ============================================================
    def as_adj_list(
        self, shuffle_d0: bool = True, shuffle_d1: bool = True
    ) -> NDArray["conn start_end coord", np.int8]:
        adj_list: NDArray["conn start_end coord", np.int8] = np.full(
            (self.n_connections, 2, 2),
            -1,
        )

        if shuffle_d1:
            flip_d1: NDArray["conn", np.float16] = np.random.rand(self.n_connections)

        # loop over all nonzero elements of the connection list
        i: int = 0
        for d, x, y in np.ndindex(self.connection_list.shape):
            if self.connection_list[d, x, y]:
                c_start: CoordTup = (x, y)
                c_end: CoordTup = (
                    x + (1 if d == 0 else 0),
                    y + (1 if d == 1 else 0),
                )
                adj_list[i, 0] = np.array(c_start)
                adj_list[i, 1] = np.array(c_end)

                # flip if shuffling
                if shuffle_d1 and (flip_d1[i] > 0.5):
                    c_s, c_e = adj_list[i, 0].copy(), adj_list[i, 1].copy()
                    adj_list[i, 0] = c_e
                    adj_list[i, 1] = c_s

                i += 1

        if shuffle_d0:
            np.random.shuffle(adj_list)

        return adj_list

    @classmethod
    def from_adj_list(
        cls,
        adj_list: NDArray["conn start_end coord", np.int8],
    ) -> "LatticeMaze":
        """create a LatticeMaze from a list of connections"""

        # Note: This has only been tested for square mazes. Might need to change some things if rectangular mazes are needed.
        grid_n: int = adj_list.max() + 1

        connection_list: NDArray["lattice_dim x y", bool] = np.zeros(
            (2, grid_n, grid_n),
            dtype=bool,
        )

        for c_start, c_end in adj_list:
            # check that exactly 1 coordinate matches
            if (c_start == c_end).sum() != 1:
                raise ValueError("invalid connection")

            # get the direction
            d: int = (c_start != c_end).argmax()

            x: int
            y: int
            # pick whichever has the lesser value in the direction `d`
            if c_start[d] < c_end[d]:
                x, y = c_start
            else:
                x, y = c_end

            connection_list[d, x, y] = True

        return LatticeMaze(
            connection_list=connection_list,
        )

    # ============================================================
    # TODO: write a to_tokens method?
    # from tokens
    # ============================================================
    @classmethod
    def from_tokens(cls, tokens: list[str]) -> "LatticeMaze":
        """create a LatticeMaze from a list of tokens"""
        if tokens[0] == SPECIAL_TOKENS["adj_list_start"]:
            adj_list_tokens = get_adj_list_tokens(tokens)
        else:
            # If we're not getting a "complete" tokenized maze, assume it's a list of coord tokens already
            adj_list_tokens = tokens

        edges: list[str] = list_split(
            adj_list_tokens,
            SPECIAL_TOKENS["adjacency_endline"],
        )

        coordinates: list[tuple[str, str]] = list()
        # what we're doing here:
        # for each edge, convert to a coord tuple and add it to the list
        # check that for each edge we have a connector, and a single token on each side
        for e in edges:
            # skip last endline
            if len(e) != 0:
                # split into start and end
                e_split: list[list] = list_split(e, SPECIAL_TOKENS["connector"])
                # print(f"{e = } {e_split = }")
                assert len(e_split) == 2, f"invalid edge: {e = } {e_split = }"
                assert all(
                    len(c) == 1 for c in e_split
                ), f"invalid edge: {e = } {e_split = }"
                coordinates.append(tuple([c[0] for c in e_split]))

        assert all(
            len(c) == 2 for c in coordinates
        ), f"invalid coordinates: {coordinates = }"

        adj_list: NDArray["conn start_end coord", np.int8] = np.full(
            (len(coordinates), 2, 2),
            -1,
        )

        for i, (c_start, c_end) in enumerate(coordinates):
            adj_list[i, 0] = np.array(coord_str_to_tuple(c_start))
            adj_list[i, 1] = np.array(coord_str_to_tuple(c_end))

        return cls.from_adj_list(adj_list)

    # ============================================================
    # to and from pixels
    # ============================================================
    def _as_pixels(self) -> Bool[np.ndarray, "x y"]:
        assert self.lattice_dim == 2, "only 2D mazes are supported"
        # Create an empty pixel grid with walls
        pixel_grid: Int[np.ndarray, "x y"] = np.full(
            (self.grid_shape[0] * 2 + 1, self.grid_shape[1] * 2 + 1), 
            False,
            dtype=np.bool_,
        )

        # Set white nodes
        pixel_grid[1::2, 1::2] = True

        # Set white connections (downward)
        for i, row in enumerate(self.connection_list[0]):
            for j, connected in enumerate(row):
                if connected:
                    pixel_grid[i * 2 + 2, j * 2 + 1] = True

        # Set white connections (rightward)
        for i, row in enumerate(self.connection_list[1]):
            for j, connected in enumerate(row):
                if connected:
                    pixel_grid[i * 2 + 1, j * 2 + 2] = True

        return pixel_grid

    def as_pixels(self) -> Bool[np.ndarray, "x y"]:
        """return a pixel grid of the maze"""
        return self._as_pixels()

    @classmethod
    def _from_pixel_grid(cls, pixel_grid: Bool[np.ndarray, "x y"]) -> tuple[Bool[np.ndarray, "lattice_dim x y"], tuple[int, int]]:
        grid_shape = (pixel_grid.shape[0] // 2, pixel_grid.shape[1] // 2)
        connection_list = np.zeros((2, *grid_shape), dtype=np.bool_)

        # Extract downward connections
        connection_list[0] = pixel_grid[2::2, 1::2]

        # Extract rightward connections
        connection_list[1] = pixel_grid[1::2, 2::2]

        return connection_list, grid_shape

    @classmethod
    def from_pixels(cls, pixel_grid: Bool[np.ndarray, "x y"]) -> "LatticeMaze":
        connection_list, grid_shape = cls._from_pixel_grid(pixel_grid)
        output: LatticeMaze = cls(connection_list=connection_list)
        assert output.grid_shape == grid_shape
        return output


    # ============================================================
    # to and from ASCII
    # ============================================================
    def _as_ascii_grid(
            self, 
            char_wall: str = "#", char_open: str = " ",
        ) -> Shaped[np.ndarray, "x y"]:
        # Get the pixel grid using to_pixels().
        pixel_grid: Bool[np.ndarray, "x y"] = self._as_pixels()

        # Replace pixel values with ASCII characters.
        ascii_grid: Shaped[np.ndarray, "x y"] = np.full(pixel_grid.shape, char_wall, dtype=str)
        ascii_grid[pixel_grid == True] = char_open

        return ascii_grid
    
    def as_ascii(
            self, 
            char_wall: str = "#", char_open: str = " ",
        ) -> str:
        """return an ASCII grid of the maze"""
        ascii_grid: Shaped[np.ndarray, "x y"] = self._as_ascii_grid(char_wall, char_open)
        return "\n".join("".join(row) for row in ascii_grid)

    @classmethod
    def from_ascii(cls, ascii_str: str, char_wall: str = "#", char_open: str = " ") -> "LatticeMaze":
        lines: list[list[str]] = ascii_str.strip().split("\n")
        ascii_grid = np.array([list(line) for line in lines], dtype=str)
        pixel_grid = (ascii_grid == char_open)
        return cls.from_pixels(pixel_grid)

@serializable_dataclass(frozen=True, kw_only=True)
class TargetedLatticeMaze(LatticeMaze):
    """A LatticeMaze with a start and end position"""
    start_pos: Coord
    end_pos: Coord

    def __post_init__(self) -> None:
        # make things numpy arrays (very jank to override frozen dataclass)
        self.__dict__["start_pos"] = np.array(self.start_pos)
        self.__dict__["end_pos"] = np.array(self.end_pos)
        # check that start and end are in bounds
        if self.start_pos[0] >= self.grid_shape[0] or self.start_pos[1] >= self.grid_shape[1]:
            raise ValueError(f"start_pos {self.start_pos} is out of bounds for grid shape {self.grid_shape}")
        if self.end_pos[0] >= self.grid_shape[0] or self.end_pos[1] >= self.grid_shape[1]:
            raise ValueError(f"end_pos {self.end_pos} is out of bounds for grid shape {self.grid_shape}")

    @classmethod
    def from_lattice_maze(
        cls,
        lattice_maze: LatticeMaze,
        start_pos: Coord,
        end_pos: Coord,
    ) -> "TargetedLatticeMaze":
        return cls(
            connection_list=lattice_maze.connection_list,
            start_pos=start_pos,
            end_pos=end_pos,
        )
    
    def as_pixels(
        self,
        wall_color: RGB = (0, 0, 0),
        open_color: RGB = (255, 255, 255),
        start_color: RGB = (0, 255, 0),
        end_color: RGB = (255, 0, 0),
    ) -> Int[np.ndarray, "x y rgb"]:

        # convert original bool pixel grid to RGB
        pixel_grid_bw: Bool[np.ndarray, "x y"] = self._as_pixels()
        pixel_grid: Int[np.ndarray, "x y rgb"] = np.full(
            (*pixel_grid_bw.shape, 3), wall_color, dtype=np.uint8
        )
        pixel_grid[pixel_grid_bw == True] = open_color

        # set start and end
        pixel_grid[self.start_pos[0]*2+1, self.start_pos[1]*2+1] = start_color
        pixel_grid[self.end_pos[0]*2+1, self.end_pos[1]*2+1] = end_color

        return pixel_grid

        
    @classmethod
    def _from_pixel_grid_with_positions(
        cls,
        pixel_grid: Int[np.ndarray, "x y rgb"],
        start_color: RGB = (0, 255, 0),
        end_color: RGB = (255, 0, 0),
    ) -> tuple[Bool[np.ndarray, "lattice_dim x y"], tuple[int, int], Coord, Coord]:
        # Convert RGB pixel grid to Bool pixel grid
        pixel_grid_bw = np.all(pixel_grid == (255, 255, 255), axis=-1)
        
        connection_list, grid_shape = cls._from_pixel_grid(pixel_grid_bw)

        # Find start and end positions in the pixel grid
        start_pos = np.argwhere(np.all(pixel_grid == start_color, axis=-1))
        end_pos = np.argwhere(np.all(pixel_grid == end_color, axis=-1))

        if start_pos.size == 0 or end_pos.size == 0:
            raise ValueError("Start and/or end positions not found in the pixel grid")

        start_pos = tuple((start_pos[0] - 1) // 2)
        end_pos = tuple((end_pos[0] - 1) // 2)

        return connection_list, grid_shape, start_pos, end_pos


    @classmethod
    def from_pixels(cls, pixel_grid: Int[np.ndarray, "x y rgb"]) -> "TargetedLatticeMaze":
        connection_list, grid_shape, start_pos, end_pos = cls._from_pixel_grid_with_positions(pixel_grid)
        return cls(connection_list=connection_list, start_pos=start_pos, end_pos=end_pos)

    
    def as_ascii(
        self,
        char_wall: str = "#",
        char_open: str = " ",
        char_start: str = "S",
        char_end: str = "E",
    ) -> str:
        """return an ASCII grid of the maze"""
        ascii_grid: Shaped[np.ndarray, "x y"] = self._as_ascii_grid(char_wall, char_open)

        # Set start and end positions
        ascii_grid[self.start_pos[0]*2+1, self.start_pos[1]*2+1] = char_start
        ascii_grid[self.end_pos[0]*2+1, self.end_pos[1]*2+1] = char_end

        return "\n".join("".join(row) for row in ascii_grid)

    @classmethod
    def from_ascii(
        cls,
        ascii_str: str,
        char_wall: str = "#",
        char_open: str = " ",
        char_start: str = "S",
        char_end: str = "E",
    ) -> "TargetedLatticeMaze":
        lines = ascii_str.strip().split("\n")
        ascii_grid = np.array([list(line) for line in lines], dtype=str)
        pixel_grid = np.full((*ascii_grid.shape, 3), (0, 0, 0), dtype=np.uint8)

        pixel_grid[ascii_grid == char_wall] = (0, 0, 0)
        pixel_grid[ascii_grid == char_open] = (255, 255, 255)
        pixel_grid[ascii_grid == char_start] = (0, 255, 0)
        pixel_grid[ascii_grid == char_end] = (255, 0, 0)

        return cls.from_pixels(pixel_grid)

@serializable_dataclass(frozen=True, kw_only=True)
class SolvedMaze(LatticeMaze):
    """Stores a maze and a solution"""

    solution: CoordArray

    # properties for similarity to TargetedLatticeMaze
    @property
    def start_pos(self) -> Coord:
        return self.solution[0]
    
    @property
    def end_pos(self) -> Coord:
        return self.solution[-1]

    # for backwards compatibility
    @property
    def maze(self) -> LatticeMaze:
        warnings.warn("maze is deprecated, SolvedMaze now inherits from LatticeMaze", DeprecationWarning)
        return self

    @classmethod
    def from_lattice_maze(cls, lattice_maze: LatticeMaze, solution: list[CoordTup]) -> "SolvedMaze":
        return cls(
            connection_list=lattice_maze.connection_list,
            solution=solution,
        )

    @classmethod
    def from_tokens(cls, tokens: list[str], data_cfg) -> "SolvedMaze":
        maze: LatticeMaze = LatticeMaze.from_tokens(tokens)
        path_tokens: list[str] = get_path_tokens(tokens)
        solution: list[str | tuple[int, int]] = decode_maze_tokens_to_coords(path_tokens, data_cfg)

        assert len(solution) > 0, f"No solution found: {solution = }"

        try:
            solution_cast: list[CoordTup] = cast(list[CoordTup], solution)
            solution_np: CoordArray = np.array(solution_cast)
        except ValueError as e:
            raise ValueError(f"Invalid solution: {solution = }") from e
        
        return cls.from_lattice_maze(lattice_maze=maze, solution=solution_np)


    def as_tuple(self) -> tuple[LatticeMaze, CoordArray]:
        warnings.warn("as_tuple is deprecated", DeprecationWarning)
        return self, self.solution