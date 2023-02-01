from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.generation.latticemaze import Coord

import numpy as np


def test_gen_dfs():
    threebythree: Coord = np.array([3, 3])
    maze = LatticeMazeGenerators.gen_dfs(threebythree)

    assert maze.connection_list.shape == (2, 3, 3)
