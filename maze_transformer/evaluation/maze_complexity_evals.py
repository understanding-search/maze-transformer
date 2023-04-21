from typing import Callable, TypeAlias

import numpy as np
from jaxtyping import Int

from maze_transformer.generation.lattice_maze import LatticeMaze, SolvedMaze
from maze_transformer.utils.utils import register_method

MAZE_COMPLEXITY_EVALS: dict[str, Callable[[SolvedMaze], float]] = dict()
MazePath: TypeAlias = Int[np.ndarray, "node x_y_pos"]


class MazeComplexityEvals:
    @staticmethod
    def count_decision_points(solution: MazePath, maze: LatticeMaze) -> int:
        """Count the number of decision points in the solution path.
        E.g if solution path is a path without any forks, this will return 0."""
        dec_points_in_path = 0
        adj_list = maze.as_adj_list()
        for node in solution:
            decision_points = np.count_nonzero(np.all(node == adj_list, axis=-1))
            if decision_points > 2:
                dec_points_in_path += 1
        return dec_points_in_path

    @register_method(MAZE_COMPLEXITY_EVALS)
    def solution_length(maze: SolvedMaze, **_) -> float:
        return len(maze.solution)

    @register_method(MAZE_COMPLEXITY_EVALS)
    def decisions_in_solution_norm(solution: MazePath, maze: LatticeMaze, **_) -> float:
        """Calculate the fraction of decisions in the solution path.

        Returns 1 if the path does not contain any decisions.
        As the number of decisions in the path increases, the fraction slowly decreases towards 0.
        """
        dec_points_in_path = MazeComplexityEvals.count_decision_points(solution, maze)
        return 1 / (1 + dec_points_in_path)

    @register_method(MAZE_COMPLEXITY_EVALS)
    def decisions_in_solution_score(
        solution: MazePath, maze: LatticeMaze, **_
    ) -> float:
        """Calculate the product of the fraction of decisions and the path length.

        Returns the path length if the path does not contain any decisions.
        As the number of decisions in the path increases at each node, the product slowly approaches 1.
        """
        dec_points_in_path = MazeComplexityEvals.count_decision_points(solution, maze)
        return 1 / (1 + dec_points_in_path) * len(solution)
