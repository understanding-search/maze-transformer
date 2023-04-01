from pathlib import Path

from maze_transformer.training.wandb_client import (
    WandbClient,
    WandbJobType,
    WandbProject,
)


def upload_dataset(name: str, path: Path):
    wandb_client = WandbClient.create(
        config={},
        project=WandbProject.UNDERSTANDING_SEARCH,
        job_type=WandbJobType.CREATE_DATASET,
    )
    wandb_client.upload_dataset(name, path)


if __name__ == "__main__":
    import fire

    fire.Fire(upload_dataset)
