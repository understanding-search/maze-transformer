import random

import numpy as np

from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.generation.latticemaze import LatticeMaze, SolvedMaze
from maze_transformer.training.mazedataset import (
    MazeDatasetConfig,
    solved_maze_to_tokens,
)


def test_solved_maze_to_tokens():
    random.seed(42)
    np.random.seed(42)
    # generate a maze
    grid_n: int = 2
    maze: LatticeMaze = LatticeMazeGenerators.gen_dfs((grid_n, grid_n))

    solution = np.array(
        maze.find_shortest_path(
            c_start=(1, 1),
            c_end=(0, 0),
        )
    )

    # generate a dataset config
    node_token_map = MazeDatasetConfig(
        name="test", grid_n=grid_n, n_mazes=10
    ).node_token_map

    solved_maze = SolvedMaze(maze, solution)

    tokens: list[str] = solved_maze_to_tokens(solved_maze, node_token_map)

    expected: list[str] = [
        "<ADJLIST_START>",
        "(1,1)",
        "<-->",
        "(0,1)",
        ";",
        "(1,1)",
        "<-->",
        "(1,0)",
        ";",
        "(0,0)",
        "<-->",
        "(1,0)",
        ";",
        "<ADJLIST_END>",
        "<ORIGIN_START>",
        "(1,1)",
        "<ORIGIN_END>",
        "<TARGET_START>",
        "(0,0)",
        "<TARGET_END>",
        "<PATH_START>",
        "(1,1)",
        "(1,0)",
        "(0,0)",
        "<PATH_END>",
    ]
    assert tokens == expected
