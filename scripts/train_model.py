from pathlib import Path

from maze_transformer.training.config import TRAINING_CONFIGS, TrainConfig
from maze_transformer.training.mazedataset import (MazeDataset,
                                                   MazeDatasetConfig)
from maze_transformer.training.training import train


def main(basepath: str, cfg_name: str = "tiny-v1"):
    train_cfg: TrainConfig = TRAINING_CONFIGS[cfg_name]

    train(
        basepath=Path(basepath),
        train_cfg=train_cfg,
        dataset_class=MazeDataset,
        data_cfg_class=MazeDatasetConfig,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
