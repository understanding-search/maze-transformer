from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Union

import wandb
from wandb.sdk.wandb_run import Run


class WandbProject(Enum):
    UNDERSTANDING_SEARCH = "understanding-search"
    INTEGRATION_TESTS = "integration-tests"


class WandbJobType(Enum):
    CREATE_DATASET = "create-dataset"
    TRAIN_MODEL = "train-model"


class WandbLogger:
    def __init__(self, run: Run):
        self._run: Run = run

    @classmethod
    def create(
        cls, config: Dict, project: Union[WandbProject, str], job_type: WandbJobType
    ) -> WandbLogger:
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        run = wandb.init(
            config=config,
            project=(project.value if isinstance(project, WandbProject) else project),
            job_type=job_type.value,
        )

        logger = WandbLogger(run)
        logger.progress(f"{config =}")
        return logger

    def upload_model(self, model_path: Path, aliases=None) -> None:
        artifact = wandb.Artifact(name=wandb.run.id, type="model")
        artifact.add_file(str(model_path))
        self._run.log_artifact(artifact, aliases=aliases)

    def upload_dataset(self, name: str, path: Path) -> None:
        artifact = wandb.Artifact(name=name, type="dataset")
        artifact.add_dir(local_path=str(path))
        self._run.log_artifact(artifact)

    def log_metric(self, data: Dict[str, Any]) -> None:
        self._run.log(data)

    def summary(self, data: Dict[str, Any]) -> None:
        self._run.summary.update(data)

    @staticmethod
    def progress(message: str) -> None:
        logging.info(message)