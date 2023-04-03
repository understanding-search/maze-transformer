import json
from pathlib import Path
from typing import Union

from maze_transformer.generation.latticemaze import SPECIAL_TOKENS
from maze_transformer.training.config import GPT_CONFIGS, TRAINING_CONFIGS, ConfigHolder
from maze_transformer.training.mazedataset import MazeDataset
from maze_transformer.training.training import TRAIN_SAVE_FILES, get_dataloader, train
from maze_transformer.training.wandb_logger import (
    WandbJobType,
    WandbLogger,
    WandbProject,
)
from maze_transformer.utils.utils import get_device


def train_model(
    basepath: str,
    wandb_project: Union[WandbProject, str],
    training_cfg: str = "tiny-v1",
    model_cfg: str = "tiny-v1",
):
    dataset = MazeDataset.disk_load(basepath, do_config=True, do_tokenized=True)

    cfg: ConfigHolder = ConfigHolder(
        dataset_cfg=dataset.cfg,
        model_cfg=GPT_CONFIGS[model_cfg],
        train_cfg=TRAINING_CONFIGS[training_cfg],
        pretrainedtokenizer_kwargs=dict(
            bos_token=SPECIAL_TOKENS["padding"],
            eos_token=SPECIAL_TOKENS["padding"],
            pad_token=SPECIAL_TOKENS["padding"],
        ),
    )

    with open(Path(basepath) / TRAIN_SAVE_FILES.config_holder, "w") as f:
        json.dump(cfg.serialize(), f, indent="\t")

    output_dir_name = TRAIN_SAVE_FILES.train_dir_format(cfg.dataset_cfg, cfg.train_cfg)
    output_path: Path = Path(basepath) / output_dir_name
    (output_path / TRAIN_SAVE_FILES.checkpoints).mkdir(parents=True)

    logger = WandbLogger.create(
        config=cfg.serialize(),
        project=wandb_project,
        job_type=WandbJobType.TRAIN_MODEL,
    )

    logger.progress("Loaded data config, initialized logger")

    logger.summary(
        dict(
            logger_cfg={
                "output_dir": str(output_path),
                "data_cfg.name": cfg.dataset_cfg.name,
                "train_cfg.name": cfg.train_cfg.name,
                "model_cfg.name": cfg.model_cfg.name,
            },
        )
    )

    dataloader = get_dataloader(dataset, cfg, logger)
    device = get_device()

    train(dataloader, cfg, logger, output_path, device)


if __name__ == "__main__":
    import fire

    fire.Fire(train_model)
