import json
import typing
import warnings
from pathlib import Path
from typing import Union

import torch
from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.dataset.configs import MAZE_DATASET_CONFIGS
from muutils.json_serialize import SerializableDataclass, serializable_dataclass
from muutils.mlutils import get_device
from torch.utils.data import DataLoader

from maze_transformer.training.config import (
    GPT_CONFIGS,
    TRAINING_CONFIGS,
    ConfigHolder,
    ZanjHookedTransformer,
)
from maze_transformer.training.train_save_files import TRAIN_SAVE_FILES
from maze_transformer.training.training import get_dataloader, train
from maze_transformer.training.wandb_logger import (
    WandbJobType,
    WandbLogger,
    WandbProject,
)


@serializable_dataclass(kw_only=True)
class TrainingResult(SerializableDataclass):
    output_path: Path
    model: ZanjHookedTransformer

    def __str__(self):
        return f"TrainingResult of training run stored at output_path='{self.output_path}', trained a model from config with name: {self.model.zanj_model_config.name}"


def train_model(
    base_path: str | Path,
    wandb_project: Union[WandbProject, str],
    cfg: ConfigHolder | None = None,
    cfg_file: str | Path | None = None,
    cfg_names: typing.Sequence[str] | None = None,
    do_generate_dataset: bool = False,
    dataset_verbose: bool = False,
    dataset: MazeDataset | None = None,
    allow_dataset_override: bool = False,
    device: torch.device | None = None,
    help: bool = False,
    **kwargs,
) -> TrainingResult:
    """specifying a location, wandb project, and config, train a model

    config be specified in one of three ways:
     - `cfg: ConfigHolder`: a ConfigHolder object (cant do this with command line)
     - `cfg_file: str|Path` path to a json file containing a config
        # TODO: allow getting config from existing saved model zanj file
     - `cfg_names: list[str]`: a 3-tuple or list of names of standard configs, optional 4th element is name for the ConfigHolder
        - dataset config names: {dataset_cfg_names}
        - model config names: {model_cfg_names}
        - train config names: {train_cfg_names}
    """
    if help:
        print(train_model.__doc__)
        return

    if device is None:
        device = get_device()

    cfg = ConfigHolder.get_config_multisource(
        cfg=cfg,
        cfg_file=cfg_file,
        cfg_names=cfg_names,
        kwargs_in=kwargs,
    )

    # set up path, save config
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    output_path = base_path / TRAIN_SAVE_FILES.model_run_dir(cfg)
    output_path = Path(output_path)
    output_path.mkdir(parents=True)
    with open(Path(output_path) / TRAIN_SAVE_FILES.config_holder, "w") as f:
        json.dump(cfg.serialize(), f, indent="\t")
    (output_path / TRAIN_SAVE_FILES.checkpoints).mkdir(parents=True)

    # set up logger
    logger: WandbLogger = WandbLogger.create(
        config=cfg.serialize(),
        project=wandb_project,
        job_type=WandbJobType.TRAIN_MODEL,
    )
    logger.progress("Initialized logger")
    logger.summary(
        dict(
            logger_cfg={
                "output_dir": output_path.as_posix(),
                "cfg.name": cfg.name,
                "data_cfg.name": cfg.dataset_cfg.name,
                "train_cfg.name": cfg.train_cfg.name,
                "model_cfg.name": cfg.model_cfg.name,
                "cfg_summary": cfg.summary(),
                "cfg": cfg.serialize(),
            },
        )
    )
    logger.progress("Summary logged, getting dataset")

    # load dataset
    if dataset is None:
        dataset = MazeDataset.from_config(
            cfg=cfg.dataset_cfg,
            do_generate=do_generate_dataset,
            local_base_path=base_path,
            verbose=dataset_verbose,
        )
    else:
        if dataset.cfg == cfg.dataset_cfg:
            logger.progress(f"passed dataset has matching config, using that")
        else:
            if allow_dataset_override:
                logger.progress(
                    f"passed dataset has different config than cfg.dataset_cfg, but allow_dataset_override is True, so using passed dataset"
                )
            else:
                datasets_cfg_diff: dict = dataset.cfg.diff(cfg.dataset_cfg)
                if datasets_cfg_diff == {
                    "applied_filters": {
                        "self": [
                            {
                                "name": "collect_generation_meta",
                                "args": (),
                                "kwargs": {},
                            }
                        ],
                        "other": [],
                    }
                }:
                    warnings.warn(
                        f"dataset has different config than cfg.dataset_cfg, but the only difference is in applied_filters, so using passed dataset. This is due to fast dataset loading collecting generation metadata for performance reasons"
                    )

                else:
                    raise ValueError(
                        f"dataset has different config than cfg.dataset_cfg, and allow_dataset_override is False",
                        f"{datasets_cfg_diff = }",
                    )

    logger.progress(f"finished getting training dataset with {len(dataset)} samples")
    # validation dataset, if applicable
    val_dataset: MazeDataset | None = None
    if cfg.train_cfg.validation_dataset_cfg is not None:
        if isinstance(cfg.train_cfg.validation_dataset_cfg, int):
            # split the training dataset
            assert len(dataset) > cfg.train_cfg.validation_dataset_cfg, (
                f"{cfg.train_cfg.validation_dataset_cfg = } "
                + f"is greater than the length of the training dataset: {len(dataset) = }"
            )
            split_dataset_sizes: tuple[int, int] = [
                len(dataset) - cfg.train_cfg.validation_dataset_cfg,
                cfg.train_cfg.validation_dataset_cfg,
            ]
            val_dataset = MazeDataset(
                cfg.dataset_cfg,
                mazes=dataset.mazes[-split_dataset_sizes[1] :],
                generation_metadata_collected=dataset.generation_metadata_collected,
            )
            dataset.mazes = dataset.mazes[: split_dataset_sizes[0]]
            dataset.update_self_config()
            val_dataset.update_self_config()
            logger.progress(
                f"got validation dataset by splitting training dataset into {len(dataset)} train and {len(val_dataset)} validation samples"
            )
        elif isinstance(cfg.train_cfg.validation_dataset_cfg, MazeDatasetConfig):
            val_dataset = MazeDataset.from_config(
                cfg=cfg.train_cfg.validation_dataset_cfg,
                do_generate=do_generate_dataset,
                local_base_path=base_path,
                verbose=dataset_verbose,
            )
            logger.progress(
                f"got custom validation dataset with {len(val_dataset)} samples"
            )

    # get dataloader and then train
    dataloader: DataLoader = get_dataloader(dataset, cfg, logger)

    logger.progress("finished dataloader, passing to train()")
    trained_model: ZanjHookedTransformer = train(
        cfg=cfg,
        dataloader=dataloader,
        logger=logger,
        output_dir=output_path,
        device=device,
        val_dataset=val_dataset,
    )

    return TrainingResult(
        output_path=output_path,
        model=trained_model,
    )


train_model.__doc__ = train_model.__doc__.format(
    dataset_cfg_names=str(list(MAZE_DATASET_CONFIGS.keys())),
    model_cfg_names=str(list(GPT_CONFIGS.keys())),
    train_cfg_names=str(list(TRAINING_CONFIGS.keys())),
)
