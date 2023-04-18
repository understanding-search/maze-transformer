import re
from pathlib import Path

import pytest
from maze_transformer.evaluation.path_evals import PathEvals

from maze_transformer.generation.constants import SPECIAL_TOKENS
from maze_transformer.training.config import GPT_CONFIGS, TRAINING_CONFIGS, ConfigHolder
from maze_transformer.training.maze_dataset import MazeDataset, MazeDatasetConfig
from maze_transformer.training.train_save_files import TRAIN_SAVE_FILES
from maze_transformer.training.training import get_dataloader, train
from maze_transformer.training.wandb_logger import WandbJobType, WandbProject
from maze_transformer.utils.utils import get_device
from scripts.create_dataset import create_dataset
from tests.helpers.stub_logger import StubLogger


@pytest.mark.usefixtures("temp_dir")
def test_train_model_pretrained_tokenizer_without_evals(temp_dir: Path):
    dataset, dataset_path = _create_dataset(temp_dir, n_mazes=5)
    cfg = _create_pretrained_tokenizer_config(dataset.cfg, batch_size=5)

    output_path = _create_output_path(cfg, dataset_path)
    logger = _create_logger(cfg)
    dataloader = get_dataloader(dataset, cfg, logger)
    device = get_device()

    train(dataloader, cfg, logger, output_path, device)

    metrics = _get_metrics(logger.logs)
    # 5 mazes in dataset and 5 per batch, so only one iteration of training loop
    assert len(metrics) == 1
    assert list(metrics[0].keys()) == ["loss"]


@pytest.mark.usefixtures("temp_dir")
def test_train_model_wrapped_tokenizer_without_evals(temp_dir: Path):
    dataset, dataset_path = _create_dataset(temp_dir, n_mazes=5)
    cfg = _create_pretrained_tokenizer_config(dataset.cfg, batch_size=5)

    output_path = _create_output_path(cfg, dataset_path)
    logger = _create_logger(cfg)
    dataloader = get_dataloader(dataset, cfg, logger)
    device = get_device()

    train(dataloader, cfg, logger, output_path, device)

    metrics = _get_metrics(logger.logs)
    assert len(metrics) == 1
    assert list(metrics[0].keys()) == ["loss"]


@pytest.mark.usefixtures("temp_dir")
def test_train_model_pretrained_tokenizer_with_evals(temp_dir: Path):
    dataset, dataset_path = _create_dataset(temp_dir, n_mazes=10)
    cfg = _create_pretrained_tokenizer_config(dataset.cfg, batch_size=5)

    output_path = _create_output_path(cfg, dataset_path)
    logger = _create_logger(cfg)
    dataloader = get_dataloader(dataset, cfg, logger)
    device = get_device()

    # fast should run every 5 mazes (1 batch), slow every 10 mazes (2 batches) 
    cfg.train_cfg.fast_eval_interval = 5
    cfg.train_cfg.slow_eval_interval = 10
    train(dataloader, cfg, logger, output_path, device)

    metrics = _get_metrics(logger.logs)
    # we should have 1 loop with fast evals and 1 loop with fast and slow
    assert len(metrics) == 2
    assert set(metrics[0].keys()) == {"loss", *PathEvals.fast.keys()}
    assert set(metrics[0].keys()) == {"loss", *PathEvals.fast.keys(), *PathEvals.slow.keys()}


@pytest.mark.usefixtures("temp_dir")
def test_train_model_wrapped_tokenizer_with_evals(temp_dir: Path):
    dataset, dataset_path = _create_dataset(temp_dir, n_mazes=10)
    cfg = _create_wrapped_tokenizer_config(dataset.cfg, batch_size=5)

    output_path = _create_output_path(cfg, dataset_path)
    logger = _create_logger(cfg)
    dataloader = get_dataloader(dataset, cfg, logger)
    device = get_device()

    # fast should run every 5 mazes (1 batch), slow every 10 mazes (2 batches) 
    cfg.train_cfg.fast_eval_interval = 5
    cfg.train_cfg.slow_eval_interval = 10
    train(dataloader, cfg, logger, output_path, device)

    metrics = _get_metrics(logger.logs)
    # we should have 1 loop with fast evals and 1 loop with fast and slow
    assert len(metrics) == 2
    assert set(metrics[0].keys()) == {"loss", *PathEvals.fast.keys()}
    assert set(metrics[0].keys()) == {"loss", *PathEvals.fast.keys(), *PathEvals.slow.keys()}


def _create_dataset(temp_dir: Path, n_mazes: int = 5, grid_n: int = 3) -> tuple[MazeDataset, Path]:
    # This dataset creation in every test is awful, will be resolved when we add dataset caching to tests
    # We shouldn't need to save and load from disk here - will resolve once we get dataset creation logic out of script
    create_dataset(path_base=str(temp_dir), n_mazes=n_mazes, grid_n=grid_n, name="test")
    dataset_path = temp_dir / "g3-n5-test"
    dataset = MazeDataset.disk_load(
        str(dataset_path), do_config=True, do_tokenized=True
    )
    dataset.cfg.seq_len_max = 32
    return dataset, dataset_path


def _create_logger(cfg: ConfigHolder) -> StubLogger:
    logger = StubLogger.create(
        config=cfg.serialize(),
        project=WandbProject.INTEGRATION_TESTS,
        job_type=WandbJobType.TRAIN_MODEL,
    )
    return logger


def _create_output_path(cfg: ConfigHolder, temp_dir: Path) -> Path:
    output_dir_name = TRAIN_SAVE_FILES.train_dir_format(cfg.dataset_cfg, cfg.train_cfg)
    output_path: Path = temp_dir / output_dir_name
    (output_path / TRAIN_SAVE_FILES.checkpoints).mkdir(parents=True)
    return output_path


def _create_pretrained_tokenizer_config(dataset_cfg: MazeDatasetConfig, batch_size: int = 5) -> ConfigHolder:
    cfg: ConfigHolder = ConfigHolder(
        dataset_cfg=dataset_cfg,
        model_cfg=GPT_CONFIGS["tiny-v1"],
        train_cfg=TRAINING_CONFIGS["tiny-v1"],
        pretrainedtokenizer_kwargs=dict(
            bos_token=SPECIAL_TOKENS["padding"],
            eos_token=SPECIAL_TOKENS["padding"],
            pad_token=SPECIAL_TOKENS["padding"],
        ),
    )

    cfg.train_cfg.dataloader_cfg["shuffle"] = False
    cfg.train_cfg.batch_size = batch_size
    return cfg


def _create_wrapped_tokenizer_config(dataset_cfg: MazeDatasetConfig, batch_size: int = 5) -> ConfigHolder:
    cfg: ConfigHolder = ConfigHolder(
        dataset_cfg=dataset_cfg,
        model_cfg=GPT_CONFIGS["tiny-v1"],
        train_cfg=TRAINING_CONFIGS["tiny-v1"],
    )
    cfg.train_cfg.dataloader_cfg["shuffle"] = False
    cfg.train_cfg.batch_size = batch_size
    return cfg


def _get_metrics(logs: list):
    metrics = [log[1][0] for log in logs if re.match("metric", log[0], re.IGNORECASE)]

    return metrics
