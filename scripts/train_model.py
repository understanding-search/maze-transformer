import json
from pathlib import Path
from typing import Union
import typing

import torch

from maze_transformer.generation.constants import SPECIAL_TOKENS
from maze_transformer.training.config import GPT_CONFIGS, TRAINING_CONFIGS, BaseGPTConfig, ConfigHolder, TrainConfig, ZanjHookedTransformer
from maze_transformer.training.maze_dataset import MazeDataset, MazeDatasetConfig
from maze_transformer.training.training import TRAIN_SAVE_FILES, get_dataloader, train
from maze_transformer.training.wandb_logger import (
    WandbJobType,
    WandbLogger,
    WandbProject,
)
from maze_transformer.utils.utils import get_device
from torch.utils.data import DataLoader
from muutils.dictmagic import kwargs_to_nested_dict


def train_model(
    output_path: str|Path,
    wandb_project: Union[WandbProject, str],
    cfg: ConfigHolder|None = None,
    cfg_file: str|Path|None = None,
    cfg_names: typing.Sequence[str]|None = None,
    do_generate_dataset: bool = False,
    **kwargs,
) -> ZanjHookedTransformer:

    cfg = ConfigHolder.get_config_cli(
        cfg=cfg,
        cfg_file=cfg_file,
        cfg_names=cfg_names,
        kwargs_in=kwargs,
    )

    # set up path, save config
    output_path = Path(output_path)
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
                "cfg": cfg.serialize(),
            },
        )
    )

    # load dataset
    dataset: MazeDataset = MazeDataset.from_config(
        cfg=cfg.dataset_cfg,
        do_generate_dataset=False,
    )
    logger.progress("loaded dataset")

    # get dataloader, device, and then train
    dataloader: DataLoader = get_dataloader(dataset, cfg, logger)
    device: torch.device = get_device()

    return train(
        cfg=cfg,
        dataloader=dataloader,
        logger=logger, 
        output_dir=output_path,
        device=device,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(train_model)
