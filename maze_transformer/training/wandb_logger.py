from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Union

import torch
import wandb
from wandb.sdk.wandb_run import Run


class WandbProject(Enum):
    UNDERSTANDING_SEARCH = "understanding-search"
    INTEGRATION_TESTS = "integration-tests"


class WandbJobType(Enum):
    CREATE_DATASET = "create-dataset"
    TRAIN_MODEL = "train-model"
    DOWNLOAD_MODEL = "download-model"


class WandbClient:
    def __init__(self, run: Run):
        self._run: Run = run

    @classmethod
    def create(
        cls, project: Union[WandbProject, str], job_type: WandbJobType, **wandb_kwargs
    ) -> WandbClient:
        run = wandb.init(
            project=(project.value if isinstance(project, WandbProject) else project),
            job_type=job_type.value,
            **wandb_kwargs
        )

        wandb_client = WandbClient(run)
        # logging.info(f"{config =}")
        return wandb_client

    def upload_model(self, model_path: Path, aliases=None) -> None:
        artifact = wandb.Artifact(name=wandb.run.id, type="model")
        artifact.add_file(str(model_path))
        self._run.log_artifact(artifact, aliases=aliases)

    def upload_dataset(self, name: str, path: Path) -> None:
        artifact = wandb.Artifact(name=name, type="dataset")
        artifact.add_dir(local_path=str(path))
        self._run.log_artifact(artifact)

    def download_from_model_registry(self, artifact_name: str, alias: str) -> Path:
        artifact = self._run.use_artifact(f"aisc-search/model-registry/{artifact_name}:{alias}", type='model')
        return Path(artifact.download())

    def log_metric(self, data: Dict[str, Any]) -> None:
        self._run.log(data)

    def summary(self, data: Dict[str, Any]) -> None:
        self._run.summary.update(data)