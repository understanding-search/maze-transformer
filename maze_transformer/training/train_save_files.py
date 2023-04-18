from datetime import datetime
from typing import Callable

from muutils.misc import freeze, sanitize_fname  # type: ignore[import]

from maze_transformer.training.config import TrainConfig
from maze_transformer.training.dataset import GPTDatasetConfig


@freeze
class TRAIN_SAVE_FILES:
    """namespace for filenames/formats for saving training data"""

    data_cfg: str = "data_config.json"
    train_cfg: str = "train_config.json"
    config_holder: str = "config.json"
    log: str = "log.jsonl"
    checkpoints: str = "checkpoints"
    train_dir_format: Callable[
        [GPTDatasetConfig, TrainConfig], str
    ] = (
        lambda d_cfg, t_cfg: f"{sanitize_fname(d_cfg.name)}_{sanitize_fname(t_cfg.name)}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    model_checkpt: Callable[[int], str] = lambda iteration: f"model.iter_{iteration}.pt"
    model_final: str = "model.final.pt"
