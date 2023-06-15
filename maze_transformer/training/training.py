from functools import partial
from pathlib import Path

import torch
from jaxtyping import Float
from maze_dataset import MazeDataset, MazeDatasetConfig, SolvedMaze
from torch.utils.data import DataLoader
from transformer_lens.HookedTransformer import SingleLoss
from zanj import ZANJ

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
) -> ZanjHookedTransformer:
    if zanj is None:
        zanj = ZANJ()
    logger.progress("Initializing model")
    model: ZanjHookedTransformer = cfg.create_model_zanj()
    logger.summary({"device": str(device), "model.device": model.cfg.device})
    logger.progress("Initializing optimizer")

    # Only the HuggingMazeTokenizer has token decoding implemented, which is required for evals
    evals_enabled = type(model.tokenizer) == HuggingMazeTokenizer
    if not evals_enabled:
        logger.progress(
            "Using a tokenizer that cannot decode. Disabling evals for this run"
        )

    optimizer: torch.optim.Optimizer = cfg.train_cfg.optimizer(
        model.parameters(),
        **cfg.train_cfg.optimizer_kwargs,
    )
    logger.summary(dict(model_n_params=model.cfg.n_params))

    model.train()
    logger.progress("Starting training")
    n_batches: int = len(dataloader)
    logger.summary({"n_batches": n_batches})

    logger.progress(
        f"will train for {n_batches} batches, {cfg.train_cfg.intervals_batches = }"
    )

    for iteration, batch in enumerate(dataloader):
        loss: SingleLoss
        logits: Float[torch.Tensor, "batch pos d_vocab"]
        logits, loss = model(batch, return_type="both")
        # Remove the last logit because it's the prediction for what comes after PATH_END (and so is meaningless)
        # Do this after computing loss because the loss_fn already ignores the last logit
        logits = logits[:, :-1, :]
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # TODO: tidy this up
        metrics: dict[str, dict[str, float | int] | float] = {"loss": float(loss)}
        if evals_enabled:
            if (
                fast_eval_interval_iters > 0
                and iteration % fast_eval_interval_iters == 0
            ):
                scores = evaluate_logits(
                    logits,
                    batch,
                    cfg,
                    model.tokenizer,
                    path_evals=PathEvals.fast,
                )
                for eval, stats in scores.items():
                    metrics[eval] = stats.summary()

            if (
                slow_eval_interval_iters > 0
                and iteration % slow_eval_interval_iters == 0
            ):
                scores = evaluate_logits(
                    logits,
                    batch,
                    cfg,
                    model.tokenizer,
                    path_evals=PathEvals.slow,
                )
                for eval, stats in scores.items():
                    metrics[eval] = stats.summary()

        print("logging metrics")
        logger.log_metric(metrics)

        if iteration % loss_interval_iters == 0:
            logger.progress(
                f"iteration {iteration}/{n_batches}: loss={loss.item():.3f}"
            )

        if iteration % loss_interval_iters == 0:
            logger.progress(
                f"iteration {iteration}/{n_batches}: loss={loss.item():.3f}"
            )

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
