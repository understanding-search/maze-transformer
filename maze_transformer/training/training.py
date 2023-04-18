from pathlib import Path

import torch
from jaxtyping import Float
from muutils.statcounter import StatCounter  # type: ignore[import]
from muutils.tensor_utils import ATensor  # type: ignore[import]
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from maze_transformer.evaluation.eval_model import evaluate_logits
from maze_transformer.evaluation.path_evals import PathEvals
from maze_transformer.training.config import ConfigHolder
from maze_transformer.training.maze_dataset import MazeDataset
from maze_transformer.training.tokenizer import HuggingMazeTokenizer
from maze_transformer.training.train_save_files import TRAIN_SAVE_FILES
from maze_transformer.training.wandb_logger import WandbLogger


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

    # TODO: These interval calculations are a bit confusing. May need some love.
    checkpoint_interval_iters: int = int(
        cfg.train_cfg.checkpoint_interval // cfg.train_cfg.batch_size
    )
    fast_eval_interval_iters: int = int(
        getattr(cfg.train_cfg, "fast_eval_interval", 0) // cfg.train_cfg.batch_size
    )
    slow_eval_interval_iters: int = int(
        getattr(cfg.train_cfg, "slow_eval_interval", 0) // cfg.train_cfg.batch_size
    )

    # TODO: check what happens in final batch where remaining mazes in dataset is less than batch size
    for iteration, batch in enumerate(dataloader):
        breakpoint()
        # compute loss
        batch_on_device: ATensor[("batch", "sequence")] = batch.type(
            dtype=torch.LongTensor
        ).to(model.cfg.device)

        loss: Float[torch.Tensor, ""]
        logits: Float[torch.Tensor, "batch pos d_vocab"]
        batch_without_last_token = batch_on_device[:, :-1]
        logits, loss = model(batch_without_last_token, return_type="both")
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
