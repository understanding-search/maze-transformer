from typing import Callable, Iterable

import numpy as np

from maze_transformer.evaluation.eval_model import ArrMazePath, MazePath
from maze_transformer.generation.latticemaze import (Coord, CoordTup,
                                                     LatticeMaze)

# pylint: disable=unused-argument

MazeEvalFunction = Callable[[LatticeMaze, MazePath, MazePath], float]
ArrMazeEvalFunction = Callable[[LatticeMaze, ArrMazePath, ArrMazePath], float]


def path_as_segments_iter(path: MazePath | ArrMazePath) -> Iterable[tuple]:
    """
    Iterate over the segments of a path.
    """
    i: int
    n_s: Coord | CoordTup
    n_e: Coord | CoordTup
    for i, n_s in enumerate(path[:-1]):
        n_e = path[i + 1]
        yield (n_s, n_e)


class MazeEvalFuncs:
    """list-of-tuples path based eval functions. first path is always the "ground truth" path"""

    @staticmethod
    def node_overlap(m: LatticeMaze, a: MazePath, b: MazePath, /) -> float:
        """number of shared nodes (any order) / total number of (unique) nodes"""

        if len(b) <= 1:
            return 0.0

        n_shared: int = 0
        a_set: set[CoordTup] = set(a)
        b_set: set[CoordTup] = set(b)

        for seg in a:
            if seg in b_set:
                n_shared += 1

        return n_shared / len(a_set)

    # @staticmethod
    # def path_len_diff(m: LatticeMaze, a: MazePath, b: MazePath, /) -> float:
    # 	"""absolute difference in path length"""

    # 	return float(len(a) - len(b))


class ArrMazeEvalFuncs:
    """array path based eval functions. first path is always the "ground truth" path"""

    @staticmethod
    def num_connections_adjacent_lattice(
        m: LatticeMaze, a: ArrMazePath, b: ArrMazePath, /
    ) -> float:
        """number of the connections in `b` which actually connect nodes that are adjacent on the lattice `m`, ignoring if they are adjacent on the maze"""

        if len(b) <= 1:
            return 0.0

        n_adj: int = 0
        for n_s, n_e in path_as_segments_iter(b):
            # print(f"{n_s = } {n_e = }")

            if (np.abs(n_s - n_e).sum() == 1).all():
                n_adj += 1

        return n_adj

    @staticmethod
    def fraction_connections_adjacent_lattice(
        m: LatticeMaze, a: ArrMazePath, b: ArrMazePath, /
    ) -> float:
        """fraction of the connections in `b` which actually connect nodes that are adjacent on the lattice `m`, ignoring if they are adjacent on the maze"""

        return ArrMazeEvalFuncs.num_connections_adjacent_lattice(m, a, b) / len(b)

    @staticmethod
    def num_connections_adjacent(
        m: LatticeMaze, a: ArrMazePath, b: ArrMazePath, /
    ) -> float:
        """number of connections in `b` which are are valid paths on the maze"""

        if len(b) <= 1:
            return 0.0

        n_connected: int = 0
        for n_s, n_e in path_as_segments_iter(b):
            if m.nodes_connected(n_s, n_e):
                n_connected += 1

        return n_connected

    @staticmethod
    def fraction_connections_adjacent(
        m: LatticeMaze, a: ArrMazePath, b: ArrMazePath, /
    ) -> float:
        """fraction of connections in `b` which are are valid paths on the maze"""

        return ArrMazeEvalFuncs.num_connections_adjacent(m, a, b) / len(b)
