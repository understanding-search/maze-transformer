from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
from muutils.misc import freeze, sanitize_fname  # type: ignore[import]
from muutils.statcounter import StatCounter  # type: ignore[import]
from muutils.tensor_utils import ATensor  # type: ignore[import]
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from maze_transformer.training.config import ConfigHolder, TrainConfig
from maze_transformer.training.dataset import GPTDatasetConfig
from maze_transformer.training.mazedataset import MazeDataset
from maze_transformer.training.wandb_logger import WandbLogger


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

    model_checkpt_zanj: Callable[[int], str] = lambda iteration: f"model.iter_{iteration}.zanj"
    model_final_zanj: str = "model.final.zanj"

def get_dataloader(
    dataset: MazeDataset, cfg: ConfigHolder, logger: WandbLogger
) -> DataLoader:
    length_stats: StatCounter = StatCounter(dataset.get_all_lengths())
    logger.summary({"dataset_seq_len_stats_summary": length_stats.summary()})
    logger.summary(
        {"dataset_seq_len_stats": length_stats.serialize(typecast=lambda x: str(x))}
    )

    logger.progress(f"Loaded {len(dataset)} sequences")
    logger.progress("Creating dataloader")
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=cfg.train_cfg.batch_size,
        **cfg.train_cfg.dataloader_cfg,
    )

    return dataloader


def train(
    dataloader: DataLoader,
    cfg: ConfigHolder,
    logger: WandbLogger,
    output_dir: Path,
    device: torch.device,
) -> None:
    logger.progress("Initializing model")
    model: HookedTransformer = cfg.create_model()
    logger.summary({"device": str(device), "model.device": model.cfg.device})

    logger.progress("Initializing optimizer")
    optimizer: torch.optim.Optimizer = cfg.train_cfg.optimizer(
        model.parameters(),
        **cfg.train_cfg.optimizer_kwargs,
    )
    logger.summary(dict(model_n_params=model.cfg.n_params))

    model.train()
    logger.progress("Starting training")
    n_batches: int = len(dataloader)
    logger.summary({"n_batches": n_batches})

    checkpoint_interval_iters: int = int(
        cfg.train_cfg.checkpoint_interval // cfg.train_cfg.batch_size
    )
    for iteration, batch in enumerate(dataloader):
        # compute loss
        batch_on_device: ATensor[("batch", "sequence")] = batch.type(
            dtype=torch.LongTensor
        ).to(model.cfg.device)

        loss = model(batch_on_device[:, :-1], return_type="loss")
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        logger.log_metric({"loss": loss})

        del loss

        if iteration % checkpoint_interval_iters == 0:
            model_save_path: Path = (
                output_dir
                / TRAIN_SAVE_FILES.checkpoints
                / TRAIN_SAVE_FILES.model_checkpt(iteration)
            )
            logger.progress(f"Saving model to {model_save_path.as_posix()}")
            torch.save(model.state_dict(), model_save_path)
            logger.upload_model(
                model_save_path, aliases=["latest", f"iter-{iteration}"]
            )

    # save the final model
    # ==================================================
    final_model_path: Path = output_dir / TRAIN_SAVE_FILES.model_final
    logger.progress(f"Saving final model to {final_model_path.as_posix()}")
    torch.save(model.state_dict(), final_model_path)
    logger.upload_model(final_model_path, aliases=["latest", "final"])

    logger.progress("Done!")
