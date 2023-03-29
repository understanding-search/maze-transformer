from pathlib import Path

import pytest

from maze_transformer.evaluation.baseline_models import RandomBaseline
from maze_transformer.evaluation.eval_model import predict_maze_paths
from maze_transformer.training.config import ConfigHolder
from maze_transformer.training.mazedataset import MazeDataset
from maze_transformer.training.training import TRAIN_SAVE_FILES
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

    base_path = Path(temp_dir / "g3-n5-test")

    dataset = MazeDataset.disk_load(path_base=base_path, do_config=True, do_tokens=True)
    model = RandomBaseline(dataset.cfg)

    max_new_tokens = 2
    paths = predict_maze_paths(
        tokens_batch=dataset.mazes_tokens,
        data_cfg=dataset.cfg,
        model=model,
        max_new_tokens=max_new_tokens,
    )

    all_coordinates = [coord for path in paths for coords in path for coord in coords]
    assert len(paths) == 5
    assert max([len(path) for path in paths]) <= max_new_tokens + 1
    assert max(all_coordinates) == grid_n - 1
