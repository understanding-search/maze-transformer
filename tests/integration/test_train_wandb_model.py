from maze_transformer.training.wandb_logger import WandbProject
from scripts.resume_wandb_run import resume_wandb_run


def test_train_wandb_model():
    wandb = resume_wandb_run(
        wandb_project=WandbProject.INTEGRATION_TESTS,
        id="ocnb1ec5"
    )
    assert True
