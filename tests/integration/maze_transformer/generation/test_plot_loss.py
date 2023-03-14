import os
import tempfile
from pathlib import Path

import pytest

from maze_transformer.evaluation.plot_loss import plot_loss
from scripts.create_dataset import create_dataset
from scripts.train_model import train_model


@pytest.fixture()
def temp_dir() -> Path:
    data_dir = tempfile.TemporaryDirectory()
    yield Path(data_dir.name)
    data_dir.cleanup()


def test_plot_loss(temp_dir):
    n_mazes = 100
    grid_n = 3
    name = "test"

    create_dataset(path_base=str(temp_dir), n_mazes=n_mazes, grid_n=grid_n, name=name)
    dataset_directory_name = f"g{grid_n}-n{n_mazes}-{name}"

    train_model(
        basepath=str(temp_dir / dataset_directory_name),
        training_cfg="tiny-v1",
        model_cfg="tiny-v1",
    )

    # hard mode: get log file name
    list_subfolders_with_paths = [
        f.path for f in os.scandir(temp_dir / dataset_directory_name) if f.is_dir()
    ]
    log_file_path = temp_dir / list_subfolders_with_paths[0] / "log.jsonl"

    plot_loss(str(log_file_path))
