from pathlib import Path

import pytest

from maze_transformer.evaluation.baseline_models import RandomBaseline
from maze_transformer.evaluation.eval_model import predict_maze_paths
from maze_transformer.generation.lattice_maze import SolvedMaze
from maze_transformer.training.config import GPT_CONFIGS, TRAINING_CONFIGS, ConfigHolder
from maze_transformer.training.maze_dataset import MazeDataset, MazeDatasetConfig
from scripts.create_dataset import create_dataset


@pytest.mark.usefixtures("temp_dir")
def test_random_baseline(temp_dir):
    # Setup will be refactored in https://github.com/orgs/AISC-understanding-search/projects/1?pane=issue&itemId=22504590
    # Disk interactions can be removed after https://github.com/AISC-understanding-search/maze-transformer/issues/113
    # First create a dataset and train a model
    grid_n = 3
    n_mazes = 5
    if not Path.exists(temp_dir / "g3-n5-test"):
        create_dataset(
            path_base=str(temp_dir), n_mazes=n_mazes, grid_n=grid_n, name="test"
        )

    cfg = ConfigHolder(
        train_cfg=TRAINING_CONFIGS["tiny-v1"],
        model_cfg=GPT_CONFIGS["tiny-v1"],
        dataset_cfg=MazeDatasetConfig(name="test", grid_n=grid_n, n_mazes=n_mazes),
    )
    base_path = Path(temp_dir / "g3-n5-test")

    dataset = MazeDataset.disk_load(path_base=base_path, do_config=True, do_tokens=True)
    unbiased_model = RandomBaseline(cfg)
    biased_model = RandomBaseline(cfg, bias=1.0)  # Always take correct path

    max_new_tokens = 15
    unbiased_paths = predict_maze_paths(
        tokens_batch=dataset.mazes_tokens,
        data_cfg=cfg.dataset_cfg,
        model=unbiased_model,
        max_new_tokens=max_new_tokens,
    )

    biased_paths = predict_maze_paths(
        tokens_batch=dataset.mazes_tokens,
        data_cfg=cfg.dataset_cfg,
        model=biased_model,
        max_new_tokens=max_new_tokens,
    )
    unbiased_coords = [
        coord for path in unbiased_paths for coords in path for coord in coords
    ]

    assert len(unbiased_paths) == 5
    assert max([len(path) for path in unbiased_paths]) <= max_new_tokens + 1
    assert max(unbiased_coords) == grid_n - 1

    for i, path in enumerate(biased_paths):
        _, solution = SolvedMaze.from_tokens(dataset.mazes_tokens[i], dataset.cfg)
        assert path == solution