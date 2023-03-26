from pathlib import Path

from maze_transformer.training.wandb_logger import (
    WandbJobType,
    WandbClient,
    WandbProject,
)


def upload_dataset(name: str, path: Path, project: str | None):
    if not project:
        project = WandbProject.UNDERSTANDING_SEARCH
        
    logger = WandbClient.create(
        config={},
        project=project,
        job_type=WandbJobType.CREATE_DATASET,
    )
    logger.upload_dataset(name, path)


if __name__ == "__main__":
    import fire

    fire.Fire(upload_dataset)
