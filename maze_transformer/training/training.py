import warnings
from functools import partial
from pathlib import Path

import torch
from jaxtyping import Float
from maze_dataset import MazeDataset, SolvedMaze
from maze_dataset.tokenization import MazeTokenizer
from muutils.mlutils import pprint_summary
from muutils.statcounter import StatCounter
from torch.utils.data import DataLoader
from transformer_lens.HookedTransformer import SingleLoss
from zanj import ZANJ

from maze_transformer.evaluation.eval_model import evaluate_model
from maze_transformer.evaluation.path_evals import PathEvals
from maze_transformer.tokenizer import HuggingMazeTokenizer
from maze_transformer.training.config import ConfigHolder, ZanjHookedTransformer
from maze_transformer.training.train_save_files import TRAIN_SAVE_FILES
from maze_transformer.training.wandb_logger import WandbLogger


def collate_batch(batch: list[SolvedMaze], maze_tokenizer: MazeTokenizer) -> list[str]:
    return [" ".join(maze.as_tokens(maze_tokenizer)) for maze in batch]


def get_dataloader(
    dataset: MazeDataset, cfg: ConfigHolder, logger: WandbLogger | None
) -> DataLoader:
    def log_progress(msg):
        # Convenience function for deciding whether to use logger or not
        if logger:
            logger.progress(msg)
        else:
            print(msg)

    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty: {len(dataset) = }")
    log_progress(f"Loaded {len(dataset)} sequences")
    log_progress("Creating dataloader")
    try:
        dataloader: DataLoader = DataLoader(
            dataset,
            collate_fn=partial(collate_batch, maze_tokenizer=cfg.maze_tokenizer),
            batch_size=cfg.train_cfg.batch_size,
            **cfg.train_cfg.dataloader_cfg,
        )
    except ValueError as e:
        raise ValueError(
            "Error creating dataloader with:",
            f"{len(dataset) = }",
            f"{cfg.train_cfg.batch_size = }",
            f"{cfg.train_cfg.dataloader_cfg = }",
            f"error: {e}",
        ) from e

    return dataloader


def train(
    cfg: ConfigHolder,
    dataloader: DataLoader,
    logger: WandbLogger,
    output_dir: Path,
    device: torch.device,
    val_dataset: MazeDataset | None = None,
    zanj: ZANJ | None = None,
    model: ZanjHookedTransformer | None = None,
) -> ZanjHookedTransformer:
    def log(msg: str | dict, log_type: str = "progress", **kwargs):
        # Convenience function to let training routine work whether or not
        # logger exists
        if logger:
            log_fn = getattr(logger, log_type)
            log_fn(msg, **kwargs)
        else:
            if type(msg) == dict:
                pprint_summary(msg)
            else:
                print(msg)

    # initialize
    # ==============================
    if zanj is None:
        zanj = ZANJ()

    # init model & optimizer
    if model is None:
        log(f"Initializing model")
        model: ZanjHookedTransformer = cfg.create_model_zanj()
        model.to(device)
    else:
        log("Using existing model")

    log({"device": str(device), "model.device": model.cfg.device}, log_type="summary")

    log("Initializing optimizer")
    optimizer: torch.optim.Optimizer = cfg.train_cfg.optimizer(
        model.parameters(),
        **cfg.train_cfg.optimizer_kwargs,
    )
    log(dict(model_n_params=model.cfg.n_params), log_type="summary")

    # add wandb run url to model
    if logger:
        model.training_records = {
            "wandb_url": logger.url,
        }

    # figure out whether to run evals, and validation dataset
    evals_enabled: bool = cfg.train_cfg.validation_dataset_cfg is not None
    if evals_enabled:
        assert (
            val_dataset is not None
        ), "val_dataset must be provided if evals are enabled"

        # Only the HuggingMazeTokenizer has token decoding implemented, which is required for evals
        if not type(model.tokenizer) == HuggingMazeTokenizer:
            warnings.warn(
                "Using a tokenizer that cannot decode. Disabling evals for this run even though TrainConfig says to enable them"
            )
            evals_enabled = False

        val_dataset_tokens: list[list[str]] = val_dataset.as_tokens(
            model.zanj_model_config.maze_tokenizer, join_tokens_individual_maze=False
        )

    # compute intervals
    n_samples: int = len(dataloader.dataset)
    n_batches: int = len(dataloader)
    intervals: dict[str, int] = cfg.train_cfg.get_intervals(
        dataset_n_samples=n_samples,
        mod_batch_size=True,
    )
    if not evals_enabled:
        intervals = {
            key: value if not key.startswith("eval") else float("inf")
            for key, value in intervals.items()
        }
    log(
        {"n_batches": n_batches, "n_samples": n_samples, "intervals": intervals},
        log_type="summary",
    )
    log(
        f"will train for {n_batches} batches, {evals_enabled=}, with intervals: {intervals}"
    )

    # TODO: add model output dir / run name to model.training_records

    # start up training
    # ==============================
    model.train()
    log("Starting training")

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
        metrics: dict[str, int | float | StatCounter] = {"loss": float(loss)}

        if evals_enabled:
            for interval_key, evals_dict in PathEvals.PATH_EVALS_MAP.items():
                if iteration % intervals[interval_key] == 0:
                    log(f"Running evals: {interval_key}")
                    scores: dict[str, StatCounter] = evaluate_model(
                        model=model,
                        dataset=val_dataset,
                        dataset_tokens=val_dataset_tokens,
                        eval_functions=evals_dict,
                        batch_size=cfg.train_cfg.batch_size,
                        max_new_tokens=cfg.train_cfg.evals_max_new_tokens,
                    )
                    metrics.update(scores)
        log(metrics, log_type="log_metric_hist")

        if iteration % intervals["print_loss"] == 0:
            log(f"iteration {iteration}/{n_batches}: loss={loss.item():.3f}")

        del loss

        # checkpoints
        # ------------------------------
        if iteration % intervals["checkpoint"] == 0:
            model_save_path: Path = (
                output_dir
                / TRAIN_SAVE_FILES.checkpoints
                / TRAIN_SAVE_FILES.model_checkpt_zanj(iteration)
            )
            log(f"Saving model checkpoint to {model_save_path.as_posix()}")
            zanj.save(model, model_save_path)
            log(
                model_save_path,
                log_type="upload_model",
                aliases=["latest", f"iter-{iteration}"],
            )

    # save the final model
    # ==============================
    final_model_path: Path = output_dir / TRAIN_SAVE_FILES.model_final_zanj
    log(f"Saving final model to {final_model_path.as_posix()}")
    zanj.save(model, final_model_path)
    log(final_model_path, log_type="upload_model", aliases=["latest", "final"])

    log("Done training!")

    return model
