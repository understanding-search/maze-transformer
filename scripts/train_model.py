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
from maze_transformer.training.training import train


def main(basepath: str, cfg_name: str = "tiny-v1"):

    dataset = MazeDataset.disk_load(basepath)

    train_cfg: TrainConfig = TRAINING_CONFIGS[cfg_name]
    model_cfg = GPT_CONFIGS[cfg_name]
    data_cfg = dataset.cfg

    output_dir = Path("models/asdfjaksdf")

    logger: Logger = Logger(
        log_path=Path(output_dir / "log.jsonl").as_posix(),
        console_print_threshold=30,
        streams=(
            LoggingStream(
                name="log_config", aliases=("cfg", "config"), default_level=40
            ),
            LoggingStream(name="train", aliases=("log_train",), default_level=50),
            LoggingStream(
                name="mem_usage",
                aliases=("traced_memory", "mem"),
                default_level=40,
                default_contents={
                    "traced_memory": (
                        lambda: dict(
                            zip(("current", "peak"), tracemalloc.get_traced_memory())
                        )
                    )
                },
            ),
        ),
    )

    logger.log("loaded data config, initialized logger")
    logger.log_config(json_serialize(cfg))
    logger.log_config(
        dict(
            logger_cfg={
                "output_dir": output_dir,
                "data_cfg.name": data_cfg.name,
                "train_cfg.name": train_cfg.name,
                "basepath": basepath,
                "model_cfg.device": model_cfg.device,
            },
            lvl=0,
        )
    )

    cfg = TopLevelConfig(dataset=data_cfg, model_config=model_cfg, train=train_cfg)

    train(dataset, cfg, logger, output_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
