import json
from pathlib import Path
from typing import Union
import typing

import torch

from maze_transformer.generation.constants import SPECIAL_TOKENS
from maze_transformer.training.config import GPT_CONFIGS, TRAINING_CONFIGS, BaseGPTConfig, ConfigHolder, TrainConfig, ZanjHookedTransformer
from maze_transformer.training.maze_dataset import MAZE_DATASET_CONFIGS, MazeDataset, MazeDatasetConfig
from maze_transformer.training.training import TRAIN_SAVE_FILES, get_dataloader, train
from maze_transformer.training.wandb_logger import (
    WandbJobType,
    WandbLogger,
    WandbProject,
)
from maze_transformer.utils.utils import get_device
from torch.utils.data import DataLoader
from muutils.dictmagic import kwargs_to_nested_dict
from muutils.misc import sanitize_fname


def train_model(
    base_path: str|Path,
    wandb_project: Union[WandbProject, str],
    cfg: ConfigHolder|None = None,
    cfg_file: str|Path|None = None,
    cfg_names: typing.Sequence[str]|None = None,
    do_generate_dataset: bool = False,
    help: bool = False,
    **kwargs,
) -> ZanjHookedTransformer:
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

    cfg = ConfigHolder.get_config_multisource(
        cfg=cfg,
        cfg_file=cfg_file,
        cfg_names=cfg_names,
        kwargs_in=kwargs,
    )

    # set up path, save config
    base_path = Path(base_path)
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
                "cfg": cfg.serialize(),
            },
        )
    )

    # load dataset
    dataset: MazeDataset = MazeDataset.from_config(
        cfg=cfg.dataset_cfg,
        do_generate=do_generate_dataset,
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
    

train_model.__doc__ = train_model.__doc__.format(
    dataset_cfg_names=str(list(MAZE_DATASET_CONFIGS.keys())),
    model_cfg_names=str(list(GPT_CONFIGS.keys())),
    train_cfg_names=str(list(TRAINING_CONFIGS.keys())),
)

if __name__ == "__main__":
    import fire

    fire.Fire(train_model)
