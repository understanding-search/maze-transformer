import tracemalloc
from pathlib import Path

from muutils.json_serialize import json_serialize
from muutils.logger import Logger, LoggingStream

from maze_transformer.training.config import (
    GPT_CONFIGS,
    TRAINING_CONFIGS,
    TopLevelConfig,
    TrainConfig,
)
from maze_transformer.training.mazedataset import MazeDataset, MazeDatasetConfig
from maze_transformer.training.training import train, setup_logger


def main(basepath: str, cfg_name: str = "tiny-v1"):

    dataset = MazeDataset.disk_load(basepath)

    # TODO: make the path some combination of model/dataset names and maybe timestamp?
    output_dir: Path = Path("models/asdfjaksdf")


    # TODO: separate names for training and model config
    cfg: TopLevelConfig = TopLevelConfig(
        dataset=dataset.cfg, 
        model_config=GPT_CONFIGS[cfg_name], 
        train=TRAINING_CONFIGS[cfg_name],
    )

    logger: Logger = setup_logger(
        output_dir = output_dir,
        config = cfg,
    )

    train(dataset, cfg, logger, output_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
