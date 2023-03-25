from pathlib import Path

from maze_transformer.training.wandb_logger import (
    WandbJobType,
    WandbLogger,
    WandbProject,
)


def upload_dataset(name: str, path: Path):
    logger = WandbLogger.create(
        config={},
        project=WandbProject.UNDERSTANDING_SEARCH,
        job_type=WandbJobType.CREATE_DATASET,
    )
    logger.upload_dataset(name, path)


if __name__ == "__main__":
    import fire

    fire.Fire(upload_dataset)
