import logging
import os
from typing import Union

import torch

from maze_transformer.training.wandb_logger import (
    WandbJobType,
    WandbClient,
    WandbProject,
)
import logging
import os
from typing import Union

import torch

from maze_transformer.training.wandb_logger import (
    WandbJobType,
    WandbClient,
    WandbProject,
)


def resume_wandb_run(
        wandb_project: Union[WandbProject, str],
        id: str
):
    wandb_client = WandbClient.create(
        project=wandb_project,
        job_type=WandbJobType.TRAIN_MODEL,
        id=id,
        resume="allow")

    model = wandb_client.download_from_model_registry("default_model", "latest")


    assert True

if __name__ == "__main__":
    import fire

    fire.Fire(resume_wandb_run)
