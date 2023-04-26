import typing

import numpy as np

from maze_transformer.generation.constants import Coord, CoordArray, CoordTup
from maze_transformer.generation.lattice_maze import LatticeMaze
from maze_transformer.utils.utils import register_method

# pylint: disable=unused-argument
MazePath = CoordArray


class PathEvalFunction(typing.Protocol):
    def __call__(
        self,
        maze: LatticeMaze | None = None,
        solution: CoordArray | None = None,
        prediction: CoordArray | None = None,
    ) -> float:
        ...


def path_as_segments_iter(path: CoordArray) -> typing.Iterable[tuple]:
    """
    Iterate over the segments of a path (ie each consecutive pair).
    """
    step_start: Coord | CoordTup
    step_end: Coord | CoordTup
    for i, step_start in enumerate(path[:-1]):
        step_end = path[i + 1]
        yield step_start, step_end


def is_adjacent(node1: Coord, node2: Coord) -> bool:
    """Check that the nodes are at most 1 step apart (diagonal steps not possible)"""
    return np.abs(node1 - node2).sum() == 1


class PathEvals:
    """array path based eval functions"""

    evals: dict[str, PathEvalFunction] = {}

    @register_method(evals)
    @staticmethod
    def node_overlap(solution: CoordArray, prediction: CoordArray, **_) -> float:
        """number of shared nodes (any order) / total number of (unique) nodes in solution"""

        solution_set = {tuple(coord) for coord in solution}
        prediction_set = {tuple(coord) for coord in prediction}

        return len(prediction_set & solution_set) / len(solution_set)

    @register_method(evals)
    @staticmethod
    def num_connections_adjacent_lattice(prediction: CoordArray, **_) -> float:
        """number of the connections in prediction which actually connect nodes that are adjacent on the lattice, ignoring if they are adjacent on the maze"""
        n_adj: float = 0.0
        for step_start, step_end in path_as_segments_iter(prediction):
            if is_adjacent(step_start, step_end):
                n_adj += 1

        return n_adj

    @register_method(evals)
    @staticmethod
    def fraction_connections_adjacent_lattice(prediction: CoordArray, **_) -> float:
        """fraction of the connections in prediction which actually connect nodes that are adjacent on the lattice, ignoring if they are adjacent on the maze"""

        return PathEvals.num_connections_adjacent_lattice(prediction) / len(prediction)

    @register_method(evals)
    @staticmethod
    def num_connections_adjacent(maze: LatticeMaze, prediction: MazePath, **_) -> float:
        """number of connections in prediction which are valid paths on the maze"""

        n_connected: int = 0
        for step_start, step_end in path_as_segments_iter(prediction):
            if maze.nodes_connected(step_start, step_end):
                n_connected += 1

        return n_connected

    @register_method(evals)
    @staticmethod
    def fraction_connections_adjacent(
        maze: LatticeMaze, prediction: CoordArray, **_
    ) -> float:
        """fraction of connections in prediction which are are valid paths on the maze"""

        num_connections: float = len(prediction) - 1.0
        return PathEvals.num_connections_adjacent(maze, prediction) / max(
            num_connections, 1.0
        )

    @register_method(evals)
    @staticmethod
    def exact_path_predicted(
        solution: CoordArray, prediction: CoordArray, **_
    ) -> float:
        """Was the maze successfully solved?"""
        return float(np.array_equal(solution, prediction))

    @register_method(evals)
    @staticmethod
    def solution_length(solution: CoordArray, **_) -> float:
        return float(len(solution))

    @register_method(evals)
    @staticmethod
    def streak_length_until_incorrect(
        solution: CoordArray,
        prediction: CoordArray,
        **_,
    ) -> float:
        """How many moves until the predicted path deviates from the solution"""
        prediction = prediction.tolist()
        solution = solution.tolist()
        streak_length: float = 0.0

        for i in range(max(len(prediction), len(solution))):
            if (
                i + 1 > len(prediction)
                or i + 1 > len(solution)
                or prediction[i] != solution[i]
            ):
                return streak_length
            else:
                streak_length += 1

        return streak_length

    @register_method(evals)
    @staticmethod
    def distance_between_end_nodes(
        solution: MazePath, prediction: MazePath, **_
    ) -> float:
        """Euclidean distance between the end nodes of the valid and predicted paths"""
        if len(prediction) <= 1:
            return 0.0

        return np.linalg.norm(solution[-1] - prediction[-1])

    @register_method(evals)
    @staticmethod
    def corner_jumps(prediction: MazePath, **_) -> float:
        """Looks for corner jumps in the predicted path. A corner jump is if the transformer predicts predicts
        (0,0) <> (1,1), instead of  (0,0) <> (0,1) <> (1,1)"""
        if len(prediction) <= 1:
            return 0.0

        pred_shift_R = prediction[1:]
        pred_shift_L = prediction[:-1]
        distance_between_nodes = pred_shift_R - pred_shift_L
        normed_distances = np.linalg.norm(distance_between_nodes, axis=1)
        return np.count_nonzero(normed_distances == np.sqrt(2))

    @register_method(evals)
    @staticmethod
    def average_predicted_step_size(prediction: MazePath, **_) -> float:
        """Returns average step size in the predicted path."""
        if len(prediction) <= 1:
            return 0.0

        pred_shift_R = prediction[1:]
        pred_shift_L = prediction[:-1]
        distance_between_nodes = pred_shift_R - pred_shift_L
        return np.linalg.norm(distance_between_nodes, axis=1).mean()
