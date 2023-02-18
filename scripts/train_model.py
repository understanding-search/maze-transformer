from pathlib import Path

import torch
from muutils.logger import Logger

from maze_transformer.training.config import GPT_CONFIGS, TRAINING_CONFIGS, ConfigHolder
from maze_transformer.training.mazedataset import MazeDataset
from maze_transformer.training.training import (
    TRAIN_SAVE_FILES,
    get_dataloader,
    setup_logger,
    train,
)


def main(basepath: str, cfg_name: str = "tiny-v1"):
    dataset = MazeDataset.disk_load(basepath, do_config=True, do_tokenized=True)

    # TODO: make the path some combination of model/dataset names and maybe timestamp?
    output_dir: Path = Path("models/asdf")
    (output_dir / TRAIN_SAVE_FILES.checkpoints).mkdir(parents=True)

    # TODO: separate names for training and model config
    cfg: ConfigHolder = ConfigHolder(
        dataset_cfg=dataset.cfg,
        model_cfg=GPT_CONFIGS[cfg_name],
        train_cfg=TRAINING_CONFIGS[cfg_name],
    )

    logger: Logger = setup_logger(
        output_dir=output_dir,
        config=cfg,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader = get_dataloader(dataset, cfg, logger)

    train(dataloader, cfg, logger, output_dir, device)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
