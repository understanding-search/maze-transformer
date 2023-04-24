from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable

import torch
from jaxtyping import Float
from muutils.misc import freeze, sanitize_fname  # type: ignore[import]
from muutils.zanj import ZANJ
from torch.utils.data import DataLoader

from maze_transformer.generation.lattice_maze import SolvedMaze
from maze_transformer.training.config import ConfigHolder, ZanjHookedTransformer
from maze_transformer.training.maze_dataset import MazeDataset, MazeDatasetConfig
from maze_transformer.training.wandb_logger import WandbLogger


@freeze
class TRAIN_SAVE_FILES:
    """namespace for filenames/formats for saving training data"""

    # old
    data_cfg: str = "data_config.json"
    train_cfg: str = "train_config.json"
    model_checkpt: Callable[[int], str] = lambda iteration: f"model.iter_{iteration}.pt"
    model_final: str = "model.final.pt"

    # keep these
    config_holder: str = "config.json"
    checkpoints: str = "checkpoints"
    log: str = "log.jsonl"
    model_checkpt_zanj: Callable[
        [int], str
    ] = lambda iteration: f"model.iter_{iteration}.zanj"
    model_final_zanj: str = "model.final.zanj"
    model_run_dir: Callable[
        [ConfigHolder], str
    ] = (
        lambda cfg: f"{sanitize_fname(cfg.name)}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )


def collate_batch(batch: list[SolvedMaze], config: MazeDatasetConfig) -> list[str]:
    # Perf could be improved by vectorizing this
    result = []
    for maze in batch:
        tokens = " ".join(maze.as_tokens(config.node_token_map))
        result.append(tokens)
    return result


def get_dataloader(
    dataset: MazeDataset, cfg: ConfigHolder, logger: WandbLogger
) -> DataLoader:
    logger.progress(f"Loaded {len(dataset)} sequences")
    logger.progress("Creating dataloader")
    dataloader: DataLoader = DataLoader(
        dataset,
        collate_fn=partial(collate_batch, config=cfg.dataset_cfg),
        batch_size=cfg.train_cfg.batch_size,
        **cfg.train_cfg.dataloader_cfg,
    )

    return dataloader


def train(
    cfg: ConfigHolder,
    dataloader: DataLoader,
    logger: WandbLogger,
    output_dir: Path,
    device: torch.device,
    zanj: ZANJ | None = None,
) -> ZanjHookedTransformer:
    if zanj is None:
        zanj = ZANJ()
    logger.progress("Initializing model")
    model: ZanjHookedTransformer = cfg.create_model_zanj()
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
        loss: Float[torch.Tensor, ""]
        logits: Float[torch.Tensor, "batch pos d_vocab"]
        logits, loss = model(batch[:1], return_type="both")
        # Remove the last logit because it's the prediction for what comes after PATH_END (and so is meaningless)
        # Do this after computing loss because the loss_fn already ignores the last logit
        logits = logits[:, :-1, :]
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        logger.log_metric({"loss": loss})

        del loss

        if iteration % checkpoint_interval_iters == 0:
            model_save_path: Path = (
                output_dir
                / TRAIN_SAVE_FILES.checkpoints
                / TRAIN_SAVE_FILES.model_checkpt_zanj(iteration)
            )
            logger.progress(f"Saving model to {model_save_path.as_posix()}")
            zanj.save(model, model_save_path)
            logger.upload_model(
                model_save_path, aliases=["latest", f"iter-{iteration}"]
            )

    # save the final model
    # ==================================================
    final_model_path: Path = output_dir / TRAIN_SAVE_FILES.model_final_zanj
    logger.progress(f"Saving final model to {final_model_path.as_posix()}")
    zanj.save(model, final_model_path)
    logger.upload_model(final_model_path, aliases=["latest", "final"])

    logger.progress("Done!")

    return model
