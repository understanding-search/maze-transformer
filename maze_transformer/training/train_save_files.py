from datetime import datetime
from typing import Callable

from muutils.misc import sanitize_fname  # type: ignore[import]

from maze_transformer.training.config import ConfigHolder


class _TRAIN_SAVE_FILES:
    """namespace for filenames/formats for saving training data"""

    # old
    data_cfg: str = "data_config.json"
    train_cfg: str = "train_config.json"
    model_checkpt: Callable[[int], str] = (
        lambda _, iteration: f"model.iter_{iteration}.pt"
    )
    model_final: str = "model.final.pt"

    # keep these
    config_holder: str = "config.json"
    checkpoints: str = "checkpoints"
    log: str = "log.jsonl"
    model_checkpt_zanj: Callable[[int], str] = (
        lambda _, iteration: f"model.iter_{iteration}.zanj"
    )
    model_final_zanj: str = "model.final.zanj"
    model_run_dir: Callable[[ConfigHolder], str] = (
        lambda _, cfg: f"{sanitize_fname(cfg.name)}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )

    @classmethod
    def __class_getitem__(cls, _):
        return cls

    def __class_getattribute__(cls, name):
        if name.startswith("__"):
            return super().__class_getattribute__(name)
        attr = cls.__dict__[name]
        return attr

    def __setattr__(self, name, value):
        raise AttributeError("TRAIN_SAVE_FILES is read-only")

    __delattr__ = __setattr__


TRAIN_SAVE_FILES = _TRAIN_SAVE_FILES()
