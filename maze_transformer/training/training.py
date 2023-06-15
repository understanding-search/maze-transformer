from functools import partial
from pathlib import Path

import torch
from jaxtyping import Float
from maze_dataset import MazeDataset, MazeDatasetConfig, SolvedMaze
from torch.utils.data import DataLoader
from transformer_lens.HookedTransformer import SingleLoss
from zanj import ZANJ
from muutils.statcounter import StatCounter

from maze_transformer.evaluation.eval_model import evaluate_logits
from maze_transformer.evaluation.path_evals import PathEvals
from maze_transformer.tokenizer import HuggingMazeTokenizer
from maze_transformer.training.config import ConfigHolder, ZanjHookedTransformer
from maze_transformer.training.train_save_files import TRAIN_SAVE_FILES
from maze_transformer.training.wandb_logger import WandbLogger


def collate_batch(batch: list[SolvedMaze], config: MazeDatasetConfig) -> list[str]:
    return [" ".join(maze.as_tokens(config.node_token_map)) for maze in batch]


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
    model: ZanjHookedTransformer | None = None,
) -> ZanjHookedTransformer:

    # initialize
    # ==============================
    if zanj is None:
        zanj = ZANJ()

    # init model & optimizer
    if model is None:
        logger.progress(f"Initializing model")
        model: ZanjHookedTransformer = cfg.create_model_zanj()
        model.to(device)
    else:
        logger.progress("Using existing model")

    logger.summary({"device": str(device), "model.device": model.cfg.device})

    logger.progress("Initializing optimizer")
    optimizer: torch.optim.Optimizer = cfg.train_cfg.optimizer(
        model.parameters(),
        **cfg.train_cfg.optimizer_kwargs,
    )
    logger.summary(dict(model_n_params=model.cfg.n_params))

    # add wandb run url to model
    model.training_records = {
        "wandb_url": logger.url,
    }
    
    # figure out whether to run evals
    # Only the HuggingMazeTokenizer has token decoding implemented, which is required for evals
    evals_enabled: bool = type(model.tokenizer) == HuggingMazeTokenizer
    if not evals_enabled:
        logger.progress(
            "Using a tokenizer that cannot decode. Disabling evals for this run"
        )
    
    # compute intervals
    n_samples: int = len(dataloader.dataset)
    n_batches: int = len(dataloader)
    intervals: dict[str, int] = cfg.train_cfg.get_intervals(
        dataset_n_samples=n_samples, 
        mod_batch_size=True,
    )
    logger.summary({"n_batches": n_batches, "n_samples": n_samples, "intervals": intervals})
    logger.progress(
        f"will train for {n_batches} batches, with intervals: {intervals}"
    )
    
    # start up training
    # ==============================
    model.train()
    logger.progress("Starting training")

    for iteration, batch in enumerate(dataloader):
        # forward pass
        # ------------------------------
        loss: SingleLoss
        logits: Float[torch.Tensor, "batch pos d_vocab"]
        logits, loss = model(batch, return_type="both")

        # backward pass
        # ------------------------------
        # Remove the last logit because it's the prediction for what comes after PATH_END (and so is meaningless)
        # Do this after computing loss because the loss_fn already ignores the last logit
        logits = logits[:, :-1, :]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # log metrics
        # ------------------------------
        metrics: dict[str, dict[str, float | int] | float] = {"loss": float(loss)}

        if evals_enabled:
            for interval_key, evals_dict in PathEvals.PATH_EVALS_MAP.items():
                if iteration % intervals[interval_key] == 0:
                    scores: dict[str, StatCounter] = evaluate_logits(
                        logits=logits,
                        batch=batch,
                        config=cfg,
                        tokenizer=model.tokenizer,
                        path_evals=evals_dict,
                    )
                    metrics.update({eval: stats.summary() for eval, stats in scores.items()})
        logger.log_metric(metrics)

        if iteration % intervals["print_loss"] == 0:
            logger.progress(
                f"iteration {iteration}/{n_batches}: loss={loss.item():.3f}"
            )

        del loss

        # checkpoints
        # ------------------------------
        if iteration % intervals["checkpoint"] == 0:
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
    # ==============================
    final_model_path: Path = output_dir / TRAIN_SAVE_FILES.model_final_zanj
    logger.progress(f"Saving final model to {final_model_path.as_posix()}")
    zanj.save(model, final_model_path)
    logger.upload_model(final_model_path, aliases=["latest", "final"])

    logger.progress("Done training!")

    return model
