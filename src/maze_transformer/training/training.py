import json
import os
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

from maze_transformer.training.config import ConfigHolder, TrainConfig
from maze_transformer.training.dataset import GPTDatasetConfig
from maze_transformer.training.mazedataset import MazeDataset


@freeze
class TRAIN_SAVE_FILES:
    """namespace for filenames/formats for saving training data"""

    data_cfg: str = "data_config.json"
    train_cfg: str = "train_config.json"
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


def setup_train(
    basepath: Path,
    train_cfg: TrainConfig,
    data_cfg_class: type = GPTDatasetConfig,
    data_cfg_fname: str = "cfg.json",
    **cfg_kwargs,
) -> tuple[GPTDatasetConfig, Logger, Path]:
    """setup for training (configs, logger, directories)

    - loads the dataset configuration from the given `basepath`
    - sets up named output directory
    - creates a logger
    - sets up training configuration
    - logs some basic information
    - returns `TrainingSetup` namedtuple

    """

    raise DeprecationWarning("this is no longer used")

    basepath = Path(basepath)

    # load the dataset config
    data_cfg_path: Path = Path(basepath) / data_cfg_fname
    with open(data_cfg_path, "r") as f:
        data_cfg: data_cfg_class = data_cfg_class.load(json.load(f))
    train_dir: str = TRAIN_SAVE_FILES.train_dir_format(data_cfg, train_cfg)

    # set up paths
    basepath_train: Path = basepath / train_dir
    os.makedirs(basepath_train, exist_ok=True)
    os.makedirs(basepath_train / TRAIN_SAVE_FILES.checkpoints, exist_ok=True)
    # store data config
    with open(basepath_train / TRAIN_SAVE_FILES.data_cfg, "w") as f:
        json.dump(json_serialize(data_cfg), f, indent="\t")

    # set up logger
    logger: Logger = Logger(
        log_path=Path(basepath_train / TRAIN_SAVE_FILES.log).as_posix(),
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

    # set up the training config
    model_cfg: OpenAIGPTConfig = train_cfg.get_gpt_config(
        **dict(
            **dict(data_cfg.gpt_config_kwargs),
            device=train_cfg.device,
        )
    )

    # store model config (after init so kwargs are correct)
    with open(basepath_train / TRAIN_SAVE_FILES.train_cfg, "w") as f:
        json.dump(json_serialize(train_cfg), f, indent="\t")

    return TrainingSetup(
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        logger=logger,
        basepath_train=basepath_train,
    )


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
