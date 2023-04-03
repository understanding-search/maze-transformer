from typing import Iterable, Optional, Protocol, TypeAlias

import numpy as np
from jaxtyping import Int

from maze_transformer.generation.constants import Coord, CoordTup
from maze_transformer.generation.latticemaze import LatticeMaze
from maze_transformer.utils.utils import register_method

# pylint: disable=unused-argument
MazePath: TypeAlias = Int[np.ndarray, "node x_y_pos"]


class PathEvalFunction(Protocol):
    def __call__(
        self,
        maze: Optional[LatticeMaze] = None,
        solution: Optional[MazePath] = None,
        prediction: Optional[MazePath] = None,
    ) -> float:
        ...


def path_as_segments_iter(path: MazePath) -> Iterable[tuple]:
    """
    Iterate over the segments of a path.
    """
    i: int
    step_start: Coord | CoordTup
    step_end: Coord | CoordTup
    for i, step_start in enumerate(path[:-1]):
        step_end = path[i + 1]
        yield step_start, step_end


class PathEvals:
    """array path based eval functions"""

    evals: dict[str, PathEvalFunction] = {}

    @register_method(evals)
    @staticmethod
    def node_overlap(solution: MazePath, prediction: MazePath, **_) -> float:
        """number of shared nodes (any order) / total number of (unique) nodes"""
        if len(prediction) <= 1:
            return 0.0

        n_shared: int = 0
        solution_set = {tuple(coord) for coord in solution}
        prediction_set = {tuple(coord) for coord in prediction}

        for coord in solution:
            if tuple(coord) in prediction_set:
                n_shared += 1
        return n_shared / len(solution_set)

    @register_method(evals)
    @staticmethod
    def num_connections_adjacent_lattice(prediction: MazePath, **_) -> float:
        """number of the connections in prediction which actually connect nodes that are adjacent on the lattice, ignoring if they are adjacent on the maze"""
        if len(prediction) <= 1:
            return 0.0

        n_adj: int = 0
        for step_start, step_end in path_as_segments_iter(prediction):
            if (np.abs(step_start - step_end).sum() == 1).all():
                n_adj += 1

        return n_adj

    @register_method(evals)
    @staticmethod
    def fraction_connections_adjacent_lattice(prediction: MazePath, **_) -> float:
        """fraction of the connections in prediction which actually connect nodes that are adjacent on the lattice, ignoring if they are adjacent on the maze"""

        return PathEvals.num_connections_adjacent_lattice(prediction) / len(prediction)

    @register_method(evals)
    @staticmethod
    def num_connections_adjacent(maze: LatticeMaze, prediction: MazePath, **_) -> float:
        """number of connections in prediction which are are valid paths on the maze"""

        if len(prediction) <= 1:
            return 0.0

        n_connected: int = 0
        for step_start, step_end in path_as_segments_iter(prediction):
            if maze.nodes_connected(step_start, step_end):
                n_connected += 1

        return n_connected

    @register_method(evals)
    @staticmethod
    def fraction_connections_adjacent(
        maze: LatticeMaze, prediction: MazePath, **_
    ) -> float:
        """fraction of connections in prediction which are are valid paths on the maze"""

        return PathEvals.num_connections_adjacent(maze, prediction) / len(prediction)
