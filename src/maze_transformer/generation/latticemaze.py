from typing import Any
from dataclasses import dataclass

import numpy as np
from muutils.tensor_utils import NDArray
from muutils.misc import list_split

# @dataclass(frozen=True, kw_only=True)
# class Maze:
# 	"""generalized maze class"""
# 	n_nodes: int


SPECIAL_TOKENS: dict[str, str] = dict(
    adjlist_start="<ADJLIST_START>",
    adjlist_end="<ADJLIST_END>",
    target_start="<TARGET_START>",
    target_end="<TARGET_END>",
    start_path="<START_PATH>",
    end_path="<END_PATH>",
    connector="<-->",
    adjacency_endline=";",
    padding="<PADDING>",
)

DIRECTIONS_MAP: NDArray[(("direction", 4), ("axes", 2)), int] = np.array(
    [
        [0, 1],  # down
        [0, -1],  # up
        [1, 1],  # right
        [1, -1],  # left
    ]
)


NEIGHBORS_MASK: NDArray["4 2", int] = np.array(
    [
        [0, 1],  # down
        [0, -1],  # up
        [1, 0],  # right
        [-1, 0],  # left
    ]
)

# print(NEIGHBORS_MASK, NEIGHBORS_MASK.dtype, NEIGHBORS_MASK.shape)

Coord = NDArray["2", np.int8]
CoordTup = tuple[int, int]
CoordArray = NDArray["points 2", np.int8]

# def get_neighbors_2d(c: Coord, maze_shape: tuple[int, int]) -> list[tuple[int, int]]:
# 	"""get the neighbors of a given coordinate"""
# 	neighbors: list[tuple[int, int]] = np.array(c) + NEIGHBORS_MASK

# 	return [
# 		[x, y]
# 		for x, y in neighbors
# 		if (0 <= x < maze_shape[0]) and (0 <= y < maze_shape[1])
# 	]


def coord_str_to_tuple(coord_str: str) -> CoordTup:
    """convert a coordinate string to a tuple"""

    stripped: str = coord_str.lstrip("(").rstrip(")")
    return tuple(int(x) for x in stripped.split(","))


@dataclass(frozen=True, kw_only=True)
class LatticeMaze:
    """lattice maze (nodes on a lattice, connections only to neighboring nodes)"""

    lattice_dim: int = 2

    """
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
    connection_list: NDArray["lattice_dim x y", bool]
    generation_meta: dict | None = None

    grid_shape = property(lambda self: self.connection_list.shape[1:])

    n_connections = property(lambda self: self.connection_list.sum())

    def as_img(self) -> NDArray["x y", bool]:
        """
        Plot an image to visualise the maze.

        Returns a matrix of side length 2n + 1 where n is the number of nodes.

        Example:
          Connection List:
          DOWN        RIGHT
          [F, T, T]   [T, T, F]
          [T, F, T]   [T, F, F]
          [F, F, F]   [T, F, F]

          Image:
                                              x x x x x x x
            N T N T N F       N - N - N       x           x
            F   T   T             |   |       x x x   x   x
            N T N F N F -->   N - N   N   --> x       x   x
            T   F   T         |       |       x   x x x   x
            N T N F N F       N - N   N       x       x   x
            F   F   F                         x x x x x x x
        """
        # set up the background
        print(self.grid_shape)
        img: NDArray["x y", bool] = np.zeros(
            (
                self.grid_shape[0] * 2 + 1,
                self.grid_shape[1] * 2 + 1,
            ),
            dtype=bool,
        )

        # fill in nodes
        img[1::2, 1::2] = True

        # fill in connections
        # print(f"{img[2:-2:2, 1::2].shape = } {self.connection_list[0, :, :-1].shape = }")
        img[2:-2:2, 1::2] = self.connection_list[0, :-1, :]
        img[1::2, 2:-2:2] = self.connection_list[1, :, :-1]

        return img

    def as_adjlist(
        self, shuffle_d0: bool = True, shuffle_d1: bool = True
    ) -> NDArray[(("conn", Any), ("start_end", 2), ("coord", 2)), np.int8]:
        adjlist: NDArray[
            (("conn", Any), ("start_end", 2), ("coord", 2)), np.int8
        ] = np.full(
            (self.n_connections, 2, 2),
            -1,
        )

        if shuffle_d1:
            flip_d1: NDArray[("conn", 1), np.float16] = np.random.rand(
                self.n_connections
            )

        # loop over all nonzero elements of the connection list
        idx: int = 0
        for d, x, y in np.ndindex(self.connection_list.shape):
            if self.connection_list[d, x, y]:
                c_start: CoordTup = (x, y)
                c_end: CoordTup = (
                    x + (1 if d == 0 else 0),
                    y + (1 if d == 1 else 0),
                )
                adjlist[idx, 0] = np.array(c_start)
                adjlist[idx, 1] = np.array(c_end)

                # flip if shuffling
                if shuffle_d1 and (flip_d1[idx] > 0.5):
                    c_s, c_e = adjlist[idx, 0].copy(), adjlist[idx, 1].copy()
                    adjlist[idx, 0] = c_e
                    adjlist[idx, 1] = c_s

                idx += 1

        if shuffle_d0:
            np.random.shuffle(adjlist)

        return adjlist

    def points_transform_to_img(self, points: CoordArray) -> CoordArray:
        """transform points to img coordinates"""
        return 2 * points + 1

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

    @classmethod
    def from_adjlist(
        cls,
        adjlist: NDArray[(("conn", Any), ("start_end", 2), ("coord", 2)), np.int8],
    ) -> "LatticeMaze":
        """create a LatticeMaze from a list of connections"""

        # TODO: this is only for square mazes
        grid_n: int = adjlist.max() + 1

        connection_list: NDArray["lattice_dim x y", bool] = np.zeros(
            (2, grid_n, grid_n),
            dtype=bool,
        )

        for c_start, c_end in adjlist:
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

    @classmethod
    def from_tokens(cls, maze_tokens: list[str]) -> "LatticeMaze":
        """create a LatticeMaze from a list of tokens"""

        edges: list[str] = list_split(
            maze_tokens,
            SPECIAL_TOKENS["adjacency_endline"],
        )

        coordinates: list[tuple[str, str]] = list()
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

        adjlist: NDArray[
            (("conn", Any), ("start_end", 2), ("coord", 2)), np.int8
        ] = np.full(
            (len(coordinates), 2, 2),
            -1,
        )

        for i, (c_start, c_end) in enumerate(coordinates):
            adjlist[i, 0] = np.array(coord_str_to_tuple(c_start))
            adjlist[i, 1] = np.array(coord_str_to_tuple(c_end))

        return cls.from_adjlist(adjlist)
