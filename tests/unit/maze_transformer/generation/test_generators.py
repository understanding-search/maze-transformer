import numpy as np

from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.generation.latticemaze import Coord


def test_gen_dfs_square():
    three_by_three: Coord = np.array([3, 3])
    maze = LatticeMazeGenerators.gen_dfs(three_by_three)

    assert maze.connection_list.shape == (2, 3, 3)


def test_gen_dfs_oblong():
    three_by_four: Coord = np.array([3, 4])
    maze = LatticeMazeGenerators.gen_dfs(three_by_four)

    assert maze.connection_list.shape == (2, 3, 4)
