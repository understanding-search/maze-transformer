import random

import numpy as np

from maze_transformer.training.mazedataset import MazeDatasetConfig
from scripts.create_dataset import get_dataset


def test_maze_dataset():
    random.seed(42)
    np.random.seed(42)

    grid_n: int = 2
    n_mazes: int = 10
    dataset_cfg = MazeDatasetConfig(name="test", grid_n=grid_n, n_mazes=n_mazes)

    dataset = get_dataset(dataset_cfg)

    assert n_mazes == len(dataset.mazes_array.indices)