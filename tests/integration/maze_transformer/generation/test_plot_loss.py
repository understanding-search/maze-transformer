import os
from pathlib import Path

import pytest

from maze_transformer.evaluation.plot_loss import plot_loss
from scripts.create_dataset import create_dataset
from scripts.train_model import train_model


@pytest.mark.usefixtures("temp_dir")
def test_plot_loss(temp_dir):
    n_mazes: int = 25
    grid_n: int = 3
    name: str = "test"

    create_dataset(path_base=str(temp_dir), n_mazes=n_mazes, grid_n=grid_n, name=name)
    dataset_directory_name: str = f"g{grid_n}-n{n_mazes}-{name}"

    train_model(
        basepath=str(temp_dir / dataset_directory_name),
        training_cfg="integration-v1",
        model_cfg="nano-v1",
    )

    # Log file name includes a timestamp so need to check the filesystem to determine the name
    list_subfolders_with_paths: list[Path] = [
        f.path for f in os.scandir(temp_dir / dataset_directory_name) if f.is_dir()
    ]
    log_file_path: Path = temp_dir / list_subfolders_with_paths[0] / "log.jsonl"

    plot_loss(str(log_file_path))

    plot_loss(str(log_file_path), raw_loss="r.")

# test_plot_loss('/tmp/bla')