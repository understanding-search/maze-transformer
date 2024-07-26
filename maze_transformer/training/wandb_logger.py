from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Union

import wandb
from muutils.statcounter import StatCounter
from wandb.sdk.wandb_run import Artifact, Run


class WandbProject(Enum):
    UNDERSTANDING_SEARCH = "understanding-search"
    DEMO_NOTEBOOKS = "demo-notebooks"
    INTEGRATION_TESTS = "integration-tests"
    __LOCAL = "local"


class WandbJobType(Enum):
    CREATE_DATASET = "create-dataset"
    TRAIN_MODEL = "train-model"


class WandbLogger:
    def __init__(self, run: Run, run_is_local: bool = False):
        self._run: Run = run
        self._run_is_local: bool = run_is_local

    @classmethod
    def create(
        cls,
        config: Dict,
        project: Union[WandbProject, str],
        job_type: WandbJobType,
        local_fallback: bool = False,
    ) -> WandbLogger:
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        run: Run
        run_is_local: bool = project == WandbProject.__LOCAL
        if not run_is_local:
            try:
                run = wandb.init(
                    config=config,
                    project=(
                        project.value if isinstance(project, WandbProject) else project
                    ),
                    job_type=job_type.value,
                )
            except Exception as e:
                if local_fallback:
                    logging.warning(
                        f"Failed to initialize wandb run, falling back to local run: {e}"
                    )
                    run_is_local = True
                else:
                    raise e

        if run_is_local:
            raise NotImplementedError("Local run not implemented yet")

        logger: WandbLogger = WandbLogger(run, run_is_local=run_is_local)
        logger.progress(f"{config =}")
        return logger

    def upload_model(self, model_path: Path, aliases=None) -> None:
        if self._run_is_local:
            raise NotImplementedError("Local run not implemented yet")
        else:
            artifact: Artifact = wandb.Artifact(name=wandb.run.id, type="model")
            artifact.add_file(str(model_path))
            self._run.log_artifact(artifact, aliases=aliases)

    def upload_dataset(self, name: str, path: Path) -> None:
        if self._run_is_local:
            raise NotImplementedError("Local run not implemented yet")
        else:
            artifact: Artifact = wandb.Artifact(name=name, type="dataset")
            artifact.add_dir(local_path=str(path))
            self._run.log_artifact(artifact)

    def log_metric(self, data: Dict[str, Any]) -> None:
        self._run.log(data)

    def log_metric_hist(self, data: dict[str, float | int | StatCounter]) -> None:
        # TODO: store the statcounters themselves somehow
        data_processed: dict[str, int | float] = dict()
        for key, value in data.items():
            if isinstance(value, StatCounter):
                # we use the mean, since then smoothing a whole bunch of evals gives us an idea of the distribution
                # data_processed[key + "-median"] = value.median()
                data_processed[key + "-mean"] = value.mean()
                # data_processed[key + "-std"] = value.std()
            else:
                data_processed[key] = value
        self._run.log(data_processed)

    def summary(self, data: Dict[str, Any]) -> None:
        self._run.summary.update(data)

    @property
    def url(self) -> str:
        return self._run.get_url()

    @staticmethod
    def progress(message: str) -> None:
        logging.info(message)
