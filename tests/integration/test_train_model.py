import pytest

from scripts.create_dataset import create_dataset
from scripts.train_model import train_model


@pytest.mark.usefixtures("temp_dir")
def test_train(temp_dir):
    create_dataset(path_base=str(temp_dir), n_mazes=5, grid_n=3, name="test")

    train_model(
        basepath=str(temp_dir / "g3-n5-test"),
        training_cfg="tiny-v1",
        model_cfg="tiny-v1",
    )
