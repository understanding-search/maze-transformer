import tempfile
from pathlib import Path

import pytest

from scripts.create_dataset import create_dataset
from scripts.train_model import train_model

"""
Tests for loading and evaluation of model

"""
@pytest.fixture()
def temp_dir() -> Path:
    data_dir = tempfile.TemporaryDirectory()
    yield Path(data_dir.name)
    data_dir.cleanup()


def test_train(temp_dir):
    create_dataset(path_base=str(temp_dir), n_mazes=5, grid_n=3, name="test")

    train_model(basepath=str(temp_dir / "g3-n5-test"), cfg_name="tiny-v1")
