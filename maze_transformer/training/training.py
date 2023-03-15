import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
from muutils.json_serialize import json_serialize  # type: ignore[import]
from muutils.logger import Logger, LoggingStream, TimerContext  # type: ignore[import]
from muutils.misc import freeze, sanitize_fname  # type: ignore[import]
from muutils.statcounter import StatCounter  # type: ignore[import]
from muutils.tensor_utils import ATensor  # type: ignore[import]
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
import wandb

from maze_transformer.training.config import ConfigHolder, TrainConfig
from maze_transformer.training.dataset import GPTDatasetConfig
from maze_transformer.training.mazedataset import MazeDataset


@freeze
class TRAIN_SAVE_FILES:
    """namespace for filenames/formats for saving training data"""

    data_cfg: str = "data_config.json"
    train_cfg: str = "train_config.json"
    config_holder: str = "config.json"
    log: str = "log.jsonl"
    checkpoints: str = "checkpoints"
    train_dir_format: Callable[
        [GPTDatasetConfig, TrainConfig], str
    ] = (
        lambda d_cfg, t_cfg: f"{sanitize_fname(d_cfg.name)}_{sanitize_fname(t_cfg.name)}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    model_checkpt: Callable[[int], str] = lambda iteration: f"model.iter_{iteration}.pt"
    model_final: str = "model.final.pt"


def setup_logger(output_path: Path, config: ConfigHolder) -> Logger:
    logger: Logger = Logger(
        log_path=Path(output_path / "log.jsonl").as_posix(),
        console_print_threshold=30,
        streams=(
            LoggingStream(
                name="log_config", aliases=("cfg", "config"), default_level=40
            ),
            LoggingStream(name="train", aliases=("log_train",), default_level=50),
            LoggingStream(
                name="mem_usage",
                aliases=("traced_memory", "mem"),
                default_level=40,
                default_contents={
                    "traced_memory": (
                        lambda: dict(
                            zip(("current", "peak"), tracemalloc.get_traced_memory())
                        )
                    )
                },
            ),
        ),
    )

    logger.log("loaded data config, initialized logger")
    logger.log_config(json_serialize(config))
    logger.log_config(
        dict(
            logger_cfg={
                "output_dir": output_path,
                "data_cfg.name": config.dataset_cfg.name,
                "train_cfg.name": config.train_cfg.name,
                "model_cfg.device": config.model_cfg.name,
            },
            lvl=0,
        )
    )

    return logger


def get_dataloader(
    dataset: MazeDataset, cfg: ConfigHolder, logger: Logger
) -> DataLoader:
    logger.log_elapsed_last()
    logger.mem_usage()

    length_stats: StatCounter = StatCounter(dataset.get_all_lengths())
    logger.log({"dataset_seq_len_stats": length_stats.summary()})
    logger.log({"dataset_seq_len_stats": length_stats.serialize()}, lvl=50)

    logger.log(f"loaded {len(dataset)} sequences", 20)
    logger.log("creating dataloader", 10)
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=cfg.train_cfg.batch_size,
        **cfg.train_cfg.dataloader_cfg,
    )

    logger.log_elapsed_last()
    logger.mem_usage()

    return dataloader


def train(
    dataloader: DataLoader,
    cfg: ConfigHolder,
    logger: Logger,
    output_dir: Path,
    device: torch.device,
) -> None:
    logger.log("load, process, and batch")
    # ==================================================
    tracemalloc.start()

    logger.log_elapsed_last()
    logger.mem_usage()

    logger.log("initialize the model and optimizer")
    # ==================================================
    logger.log("initializing model", 10)
    model: HookedTransformer = cfg.create_model()
    wandb.watch(model, model.loss_fn, log="all", log_freq=10)

    logger.log_elapsed_last()
    logger.mem_usage()
    logger.log({"device": device, "model.device": model.cfg.device}, 20)

    logger.log("initializing optimizer", 10)
    optimizer: torch.optim.Optimizer = cfg.train_cfg.optimizer(
        model.parameters(),
        **cfg.train_cfg.optimizer_kwargs,
    )
    logger.log_elapsed_last()
    logger.mem_usage()
    logger.log(dict(model_n_params=model.cfg.n_params), 20)

    # train the model
    # ==================================================
    if cfg.train_cfg.epochs > 1:
        raise NotImplementedError(
            "multiple epochs not implemented, get more data instead"
        )

    model.train()
    logger.log("starting training")
    n_batches: int = len(dataloader)
    logger.log({"n_batches": n_batches}, 10)

    n_sequences: int
    print_loss_interval_iters: int = int(
        cfg.train_cfg.print_loss_interval // cfg.train_cfg.batch_size
    )
    checkpoint_interval_iters: int = int(
        cfg.train_cfg.checkpoint_interval // cfg.train_cfg.batch_size
    )
    for iteration, batch in enumerate(dataloader):
        # compute loss
        with TimerContext() as timer_loss:
            batch_on_device: ATensor[("batch", "sequence")] = batch.type(
                dtype=torch.LongTensor
            ).to(model.cfg.device)
            # logger.tensor_dims({
            # 	"batch_on_device.shape" : batch_on_device.shape,
            # 	"batch_on_device.dtype" : str(batch_on_device.dtype),
            # 	"batch_on_device.device" : str(batch_on_device.device),
            # }, lvl = 20)

            loss = model(batch_on_device[:, :-1], return_type="loss")
            loss.backward()

        # optimize
        with TimerContext() as timer_optim:
            optimizer.step()
            optimizer.zero_grad()

        # logging
        n_sequences = iteration * cfg.train_cfg.batch_size
        log_data: dict[str, Any] = json_serialize(
            {
                "iter": iteration,
                "loss": loss,
                # "train/grad_norm": output.grad_norm,
                "n_sequences": n_sequences,
                "time_current": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "timer_loss": round(timer_loss.elapsed_time, 6),
                "timer_optim": round(timer_optim.elapsed_time, 6),
            }
        )

        del loss

        logger.train(
            log_data,
            lvl=50,
            console_print=(
                (iteration % print_loss_interval_iters == 0)
                or (iteration % checkpoint_interval_iters == 0)
            ),
        )
        wandb.log(log_data, step=iteration)

        if iteration % checkpoint_interval_iters == 0:
            model_save_path: Path = (
                output_dir
                / TRAIN_SAVE_FILES.checkpoints
                / TRAIN_SAVE_FILES.model_checkpt(iteration)
            )
            logger.saving(f"saving model to {model_save_path.as_posix()}", 10)
            torch.save(model.state_dict(), model_save_path)
            logger.log_elapsed_last(stream="saving")

    # save the final model
    # ==================================================
    final_model_path: str = output_dir / TRAIN_SAVE_FILES.model_final
    logger.saving(f"saving final model to {final_model_path.as_posix()}", 10)
    torch.save(model.state_dict(), final_model_path)
    logger.log_elapsed_last(stream="saving")

    logger.log("done!", 10)
