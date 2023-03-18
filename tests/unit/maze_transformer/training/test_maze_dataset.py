import random

import numpy as np
import torch

from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.generation.latticemaze import CoordTup, LatticeMaze
from maze_transformer.training.dataset import IndexedArray
from maze_transformer.training.mazedataset import MazeDatasetConfig, maze_to_tokens
from maze_transformer.training.tokenizer import MazeTokenizer


def test_indexed_array():
    indexed_array = IndexedArray.from_sequences([
        torch.tensor(
            [1, 2, 3, 4, 5],
            dtype=torch.int16,
        ) for _ in range(10)
    ])

    indexed_array.get_len(9)

    assert True


def test_maze_to_tokens():
    random.seed(42)
    np.random.seed(42)
    # generate a maze
    grid_n: int = 2
    maze: LatticeMaze = LatticeMazeGenerators.gen_dfs((grid_n, grid_n))

    # generate a maze tokenizer
    c_start: CoordTup = (1, 1)
    c_end: CoordTup = (0, 0)
    maze_tokenizer = MazeTokenizer(
        maze=maze,
        solution=np.array(
            maze.find_shortest_path(
                c_start=c_start,
                c_end=c_end,
            )
        ),
    )

    # generate a dataset config
    n_mazes: int = 10
    dataset_cfg = MazeDatasetConfig(name="test", grid_n=grid_n, n_mazes=n_mazes)

    tokens: list[str] = maze_to_tokens(maze_tokenizer, dataset_cfg.node_token_map)
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
