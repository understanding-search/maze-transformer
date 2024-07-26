import re
from copy import deepcopy
from pathlib import Path

import pytest
from maze_dataset import MazeDataset, MazeDatasetConfig
from muutils.mlutils import get_device

from maze_transformer.evaluation.path_evals import PathEvals
from maze_transformer.test_helpers.stub_logger import StubLogger
from maze_transformer.training.config import GPT_CONFIGS, TRAINING_CONFIGS, ConfigHolder
from maze_transformer.training.train_save_files import TRAIN_SAVE_FILES
from maze_transformer.training.training import get_dataloader, train
from maze_transformer.training.wandb_logger import WandbJobType, WandbProject


def test_train_save_files_frozen():
    with pytest.raises(AttributeError):
        TRAIN_SAVE_FILES.data_cfg = "new"


@pytest.mark.usefixtures("temp_dir")
def test_train_model_without_evals(temp_dir: Path):
    dataset = _create_dataset()
    cfg = _create_tokenizer_config(dataset.cfg, batch_size=5)

    output_path = _create_output_path(cfg, temp_dir)
    logger = _create_logger(cfg)
    dataloader = get_dataloader(dataset, cfg, logger)
    device = get_device()
    cfg.train_cfg.validation_dataset_cfg = None

    train(
        dataloader=dataloader,
        cfg=cfg,
        logger=logger,
        output_dir=output_path,
        device=device,
    )

    metrics = _get_metrics(logger.logs)
    assert len(metrics) == 2
    assert list(metrics[0].keys()) == ["loss"]


@pytest.mark.usefixtures("temp_dir")
def test_train_model_with_evals(temp_dir: Path):
    dataset = _create_dataset()
    cfg = _create_tokenizer_config(dataset.cfg, batch_size=5)

    output_path = _create_output_path(cfg, temp_dir)
    logger = _create_logger(cfg)
    dataloader = get_dataloader(dataset, cfg, logger)
    device = get_device()

    # fast should run every 5 mazes (1 batch), slow every 10 mazes (2 batches)
    cfg.train_cfg.intervals = dict(
        print_loss=1,
        checkpoint=10,
        eval_fast=5,
        eval_slow=10,
    )
    cfg.train_cfg.intervals_count = None
    cfg.train_cfg.validation_dataset_cfg = deepcopy(cfg.dataset_cfg)
    val_dataset: MazeDataset = MazeDataset.from_config(
        cfg.train_cfg.validation_dataset_cfg,
    )

    train(
        dataloader=dataloader,
        cfg=cfg,
        logger=logger,
        output_dir=output_path,
        device=device,
        val_dataset=val_dataset,
    )

    metrics = _get_metrics(logger.logs)

    # we should have 1 loop with fast evals and 1 loop with fast and slow
    assert len(metrics) == 2
    assert set(metrics[0].keys()) == {"loss", *PathEvals.fast.keys()}
    assert set(metrics[0].keys()) == {
        "loss",
        *PathEvals.fast.keys(),
        *PathEvals.slow.keys(),
    }


def _create_dataset(n_mazes: int = 10, grid_n: int = 3) -> MazeDataset:
    dataset_cfg: MazeDatasetConfig = MazeDatasetConfig(
        name="test", n_mazes=n_mazes, grid_n=grid_n
    )
    dataset = MazeDataset.from_config(dataset_cfg)
    # dataset.cfg.seq_len_max = 32
    # TODO(@mivanit): the above line caused me much pain. setting the sequence length in the tokenizer to below the length of the actual sequence passed causes horrible things to happen in `predict_maze_paths()`
    return dataset


def _create_logger(cfg: ConfigHolder) -> StubLogger:
    logger = StubLogger.create(
        config=cfg.serialize(),
        project=WandbProject.INTEGRATION_TESTS,
        job_type=WandbJobType.TRAIN_MODEL,
    )
    return logger


def _create_output_path(cfg: ConfigHolder, temp_dir: Path) -> Path:
    output_dir_name = TRAIN_SAVE_FILES.model_run_dir(cfg)
    output_path: Path = temp_dir / output_dir_name
    (output_path / TRAIN_SAVE_FILES.checkpoints).mkdir(parents=True)
    return output_path


def _create_tokenizer_config(
    dataset_cfg: MazeDatasetConfig, batch_size: int = 5
) -> ConfigHolder:
    cfg: ConfigHolder = ConfigHolder(
        dataset_cfg=dataset_cfg,
        model_cfg=GPT_CONFIGS["tiny-v1"],
        train_cfg=TRAINING_CONFIGS["tiny-v1"],
    )
    cfg.train_cfg.dataloader_cfg["shuffle"] = False
    cfg.train_cfg.batch_size = batch_size
    return cfg


def _get_metrics(logs: list):
    # for x in logs:
    #     print(x)
    metrics = [log[1][0] for log in logs if re.match("metric", log[0], re.IGNORECASE)]

    return metrics
