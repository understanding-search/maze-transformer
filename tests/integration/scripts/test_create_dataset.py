import os
import tempfile
from pathlib import Path

import pytest

from maze_transformer.training.mazedataset import MazeDataset
from scripts.create_dataset import create_dataset


@pytest.fixture
def temp_dir() -> Path:
    data_dir = tempfile.TemporaryDirectory()
    yield Path(data_dir.name)
    data_dir.cleanup()


def test_create_expected_files_and_directories(temp_dir):
    n_mazes = 5
    grid_n = 3
    name = "test"

    create_dataset(path_base=str(temp_dir), n_mazes=n_mazes, grid_n=grid_n, name=name)
    dataset_directory_name = f"g{grid_n}-n{n_mazes}-{name}"
    files = MazeDataset.DISK_SAVE_FILES
    file_names = {
        value for attr, value in vars(files).items() if not attr.startswith("__")
    }

    assert os.path.isdir(os.path.join(temp_dir, dataset_directory_name))
    for file_name in file_names:
        assert os.path.isfile(os.path.join(temp_dir, dataset_directory_name, file_name))


def test_invalid_n_mazes_values(temp_dir):
    with pytest.raises(ValueError):
        create_dataset(path_base=str(temp_dir), n_mazes=-1, grid_n=3, name="test")


def test_invalid_grid_n_values(temp_dir):
    with pytest.raises(ValueError):
        create_dataset(path_base=str(temp_dir), n_mazes=5, grid_n=-1, name="test")


def test_invalid_path(temp_dir):
    create_dataset(path_base=str(temp_dir), n_mazes=5, grid_n=3, name="test")

    with pytest.raises(FileExistsError):
        create_dataset(path_base=str(temp_dir), n_mazes=5, grid_n=3, name="test")
