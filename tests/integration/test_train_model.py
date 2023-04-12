import pytest
from maze_transformer.training.config import ZanjHookedTransformer

from maze_transformer.training.wandb_logger import WandbProject
from scripts.create_dataset import create_dataset
from scripts.train_model import train_model


@pytest.mark.usefixtures("temp_dir")
def test_train_model(temp_dir):
    model: ZanjHookedTransformer = train_model(
        output_path=temp_dir,
        wandb_project=WandbProject.INTEGRATION_TESTS,
        cfg_names=("test-g3-n5-a_dfs", "nano-v1", "integration-v1"),
        do_generate_dataset=True,
    )

    assert isinstance(model, ZanjHookedTransformer)

