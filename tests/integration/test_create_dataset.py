import os
import tempfile
from pathlib import Path

import pytest

from muutils.zanj import ZANJ
from maze_transformer.training.maze_dataset import MazeDataset, MazeDatasetConfig
from scripts.create_dataset import create_dataset


@pytest.fixture
def temp_dir() -> Path:
    data_dir = tempfile.TemporaryDirectory()
    yield Path(data_dir.name)
    data_dir.cleanup()


def test_generate_mazedataset():
    m: MazeDataset = MazeDataset.from_config(
        MazeDatasetConfig(name="test", grid_n=3, n_mazes=5),
        load_local=False,
        do_download=False,
        save_local=False,
    )

    assert len(m.mazes) == 5

def test_load_save_mazedataset_auto():
    m: MazeDataset = MazeDataset.from_config(
        MazeDatasetConfig(name="my_funky_dataset", grid_n=3, n_mazes=5),
        load_local=False,
        do_download=False,
        save_local=True,
        local_base_path=Path(str(temp_dir)),
    )

    m2: MazeDataset = MazeDataset.from_config(
        MazeDatasetConfig(name="my_funky_dataset", grid_n=3, n_mazes=5),
        load_local=True,
        do_download=False,
        do_generate=False,
        save_local=False,
        local_base_path=Path(str(temp_dir)),
    )

    assert len(m.mazes) == 5
    assert len(m2.mazes) == 5

    assert m.cfg == m2.cfg
    assert all([m1 == m2 for m1, m2 in zip(m.mazes, m2.mazes)])

def test_load_save_mazedataset_manual():
    m: MazeDataset = MazeDataset.from_config(
        MazeDatasetConfig(name="my_funky_dataset", grid_n=3, n_mazes=5),
        load_local=False,
        do_download=False,
        save_local=True,
        local_base_path=Path(str(temp_dir)),
    )

    m2_fname: Path = Path(str(temp_dir)) / m.cfg.to_fname()

    m2: MazeDataset = MazeDataset.load(ZANJ().read(m2_fname))

    assert len(m.mazes) == 5
    assert len(m2.mazes) == 5

    assert m.cfg == m2.cfg
    assert all([m1 == m2 for m1, m2 in zip(m.mazes, m2.mazes)])

@pytest.mark.skip(reason="old maze dataset interface")
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

@pytest.mark.skip(reason="old maze dataset interface")
def test_invalid_n_mazes_values(temp_dir):
    with pytest.raises(ValueError):
        create_dataset(path_base=str(temp_dir), n_mazes=-1, grid_n=3, name="test")

@pytest.mark.skip(reason="old maze dataset interface")
def test_invalid_grid_n_values(temp_dir):
    with pytest.raises(ValueError):
        create_dataset(path_base=str(temp_dir), n_mazes=5, grid_n=-1, name="test")

@pytest.mark.skip(reason="old maze dataset interface")
def test_invalid_path(temp_dir):
    create_dataset(path_base=str(temp_dir), n_mazes=5, grid_n=3, name="test")

    with pytest.raises(FileExistsError):
        create_dataset(path_base=str(temp_dir), n_mazes=5, grid_n=3, name="test")
