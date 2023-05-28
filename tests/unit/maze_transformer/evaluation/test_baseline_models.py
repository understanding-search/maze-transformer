import numpy as np
import pytest

from maze_transformer.dataset.maze_dataset import MazeDataset, MazeDatasetConfig
from maze_transformer.evaluation.baseline_models import RandomBaseline
from maze_transformer.evaluation.eval_model import predict_maze_paths
from maze_transformer.generation.lattice_maze import SolvedMaze
from maze_transformer.training.config import GPT_CONFIGS, TRAINING_CONFIGS, ConfigHolder


@pytest.mark.usefixtures("temp_dir")
def test_random_baseline(temp_dir):
    # Setup will be refactored in https://github.com/orgs/AISC-understanding-search/projects/1?pane=issue&itemId=22504590
    # Disk interactions can be removed after https://github.com/AISC-understanding-search/maze-transformer/issues/113
    # First create a dataset and train a model
    cfg: ConfigHolder = ConfigHolder(
        train_cfg=TRAINING_CONFIGS["test-v1"],
        model_cfg=GPT_CONFIGS["tiny-v1"],
        dataset_cfg=MazeDatasetConfig(name="test", grid_n=3, n_mazes=5),
    )

    dataset: MazeDataset = MazeDataset.from_config(cfg.dataset_cfg, save_local=False)
    unbiased_model: RandomBaseline = RandomBaseline(cfg)
    biased_model: RandomBaseline = RandomBaseline(cfg, bias=1.0)  # Always take correct path

    max_new_tokens: int = 15
    dataset_tokens: list[list[str]] = dataset.as_tokens(join_tokens_individual_maze=False)
    # print(f"{dataset_tokens = }")

    print("="*100)
    print("predicting unbiased paths")
    print("="*100)
    unbiased_paths = predict_maze_paths(
        tokens_batch=dataset_tokens,
        data_cfg=cfg.dataset_cfg,
        model=unbiased_model,
        max_new_tokens=max_new_tokens,
    )

    print("="*100)
    print("predicting biased paths")
    print("="*100)
    biased_paths = predict_maze_paths(
        tokens_batch=dataset_tokens,
        data_cfg=cfg.dataset_cfg,
        model=biased_model,
        max_new_tokens=max_new_tokens,
    )
    unbiased_coords = [
        coord 
        for path in unbiased_paths 
        for coords in path 
        for coord in coords
    ]

    assert len(unbiased_paths) == len(dataset)
    assert max([len(path) for path in unbiased_paths]) <= max_new_tokens + 1
    assert max(unbiased_coords) <= cfg.dataset_cfg.grid_n - 1

    for i, path in enumerate(biased_paths):
        solved_maze: SolvedMaze = dataset[i]
        assert np.all(
            np.array(path) == np.array(solved_maze.solution)
        ), f"Path {i} is not the solution: {path} != {solved_maze.solution.tolist()}\n{solved_maze.as_ascii()}\n{solved_maze}"
