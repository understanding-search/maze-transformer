import types
import typing

import numpy as np
from jaxtyping import Int

from maze_transformer.generation.constants import Coord, CoordTup
from maze_transformer.generation.latticemaze import LatticeMaze

# pylint: disable=unused-argument
MazePath: typing.TypeAlias = Int[np.ndarray, "node x_y_pos"]
PathEvalFunction = typing.Callable[[LatticeMaze, MazePath, MazePath], float]


def path_as_segments_iter(path: MazePath) -> typing.Iterable[tuple]:
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
    """array path based eval functions. first path is always the "ground truth" path

    if you add a util function that isnt a path eval function, add it to the EXCLUDED_MEMBERS list
    """

    _EXCLUDED_MEMBERS: typing.Sequence[str] = tuple(
        [
            "all_functions",
            # the following two are excluded automatically because they start with "_"
            # this comment should be left in to why they are absent from this list
            # "_EXCLUDED_MEMBERS",
            # "_assert_is_eval_function",
        ]
    )

    @staticmethod
    def _assert_is_eval_function(func: typing.Callable) -> None:
        """asserts that the function is a valid **path eval function**"""
        assert isinstance(func, typing.Callable), "func must be callable"
        assert isinstance(func, types.FunctionType), "func must be a static function"

        hints: dict[str, type] = typing.get_type_hints(func)
        assert hints["return"] is float, "path evals must return a float"
        assert hints["maze"] is LatticeMaze, "maze must be a LatticeMaze"
        assert hints["solution"] is MazePath, "solution must be a MazePath"
        assert hints["prediction"] is MazePath, "prediction must be a MazePath"

    @classmethod
    def all_functions(cls) -> dict[str, PathEvalFunction]:
        """returns a dict of all the path eval functions

        will throw except if any non_excluded member is not a valid path eval function
        """
        output: dict[str, PathEvalFunction] = dict()
        for name, func in cls.__dict__.items():
            if not name.startswith("_") and name not in cls._EXCLUDED_MEMBERS:
                cls._assert_is_eval_function(func)
                output[name] = func

        return output

    @staticmethod
    def node_overlap(
        maze: LatticeMaze,
        solution: MazePath,
        prediction: MazePath,
    ) -> float:
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

    @staticmethod
    def num_connections_adjacent_lattice(
        maze: LatticeMaze,
        solution: MazePath,
        prediction: MazePath,
    ) -> float:
        """number of the connections in prediction which actually connect nodes that are adjacent on the lattice, ignoring if they are adjacent on the maze"""
        if len(prediction) <= 1:
            return 0.0

        n_adj: int = 0
        for step_start, step_end in path_as_segments_iter(prediction):
            if (np.abs(step_start - step_end).sum() == 1).all():
                n_adj += 1

        return n_adj

    @staticmethod
    def fraction_connections_adjacent_lattice(
        maze: LatticeMaze,
        solution: MazePath,
        prediction: MazePath,
    ) -> float:
        """fraction of the connections in prediction which actually connect nodes that are adjacent on the lattice, ignoring if they are adjacent on the maze"""

        return PathEvals.num_connections_adjacent_lattice(
            maze, solution, prediction
        ) / len(prediction)

    @staticmethod
    def num_connections_adjacent(
        maze: LatticeMaze,
        solution: MazePath,
        prediction: MazePath,
    ) -> float:
        """number of connections in prediction which are are valid paths on the maze"""

        if len(prediction) <= 1:
            return 0.0

        n_connected: int = 0
        for step_start, step_end in path_as_segments_iter(prediction):
            if maze.nodes_connected(step_start, step_end):
                n_connected += 1

        return n_connected

    @staticmethod
    def fraction_connections_adjacent(
        maze: LatticeMaze,
        solution: MazePath,
        prediction: MazePath,
    ) -> float:
        """fraction of connections in prediction which are are valid paths on the maze"""

        return PathEvals.num_connections_adjacent(maze, solution, prediction) / len(
            prediction
        )
