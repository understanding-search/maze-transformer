import pytest

from maze_transformer.training.wandb_logger import WandbProject
from scripts.create_dataset import create_dataset
from scripts.train_model import train_model


@pytest.mark.usefixtures("temp_dir")
def test_train_model(temp_dir):
    create_dataset(path_base=str(temp_dir), n_mazes=5, grid_n=3, name="test")

    train_model(
        basepath=str(temp_dir / "g3-n5-test"),
        wandb_project=WandbProject.INTEGRATION_TESTS,
        training_cfg="tiny-v1",
        model_cfg="tiny-v1",
    )
