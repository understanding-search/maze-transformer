import numpy as np

from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.generation.lattice_maze import Coord


def test_gen_dfs_square():
    three_by_three: Coord = np.array([3, 3])
    maze = LatticeMazeGenerators.gen_dfs(three_by_three)

    assert maze.connection_list.shape == (2, 3, 3)


def test_gen_dfs_oblong():
    three_by_four: Coord = np.array([3, 4])
    maze = LatticeMazeGenerators.gen_dfs(three_by_four)

    assert maze.connection_list.shape == (2, 3, 4)


def test_gen_dfs_with_solution():
    three_by_three: Coord = np.array([3, 3])
    maze, solution = LatticeMazeGenerators.gen_dfs_with_solution(
        three_by_three
    ).as_tuple()

    assert maze.connection_list.shape == (2, 3, 3)
    assert len(solution[0]) == 2


def test_wilson_generation():
    maze = LatticeMazeGenerators.gen_wilson(np.array([2, 2]))
    assert maze.connection_list.shape == (2, 2, 2)
