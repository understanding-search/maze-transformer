import os
from typing import Any

import torch
import wandb

from maze_transformer.training.wandb_logger import (
    WandbJobType,
    WandbClient,
    WandbProject,
)


def download_model(
        name: str = WandbProject.UNDERSTANDING_SEARCH,
        project: str | None = None
) -> Any:
    wandb_client = WandbClient.create(
        project=project,
        job_type=WandbJobType.DOWNLOAD_MODEL,
    )

    artifact_dir = wandb_client.download_from_model_registry(name, "latest")
    files = os.listdir(artifact_dir)
    _model_path = files[0]
    model_path = artifact_dir/_model_path

    # config = wandb_client.
    return torch.load(model_path)
    # config =

    # logging.info(f"{config =}")



if __name__ == "__main__":
    import fire

    fire.Fire(download_model)
