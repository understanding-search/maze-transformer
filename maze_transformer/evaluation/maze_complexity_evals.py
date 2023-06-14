import typing

from maze_transformer.generation.lattice_maze import SolvedMaze
from muutils.mlutils import register_method

MAZE_COMPLEXITY_EVALS: dict[str, typing.Callable[[SolvedMaze], float]] = dict()


class MazeComplexityEvals:
    @register_method(MAZE_COMPLEXITY_EVALS)
    def solution_length(maze: SolvedMaze) -> float:
        return len(maze.solution)
