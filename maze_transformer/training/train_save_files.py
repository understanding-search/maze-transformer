from datetime import datetime
from typing import Callable

from muutils.misc import freeze, sanitize_fname  # type: ignore[import]


@freeze
class TRAIN_SAVE_FILES:
    """namespace for filenames/formats for saving training data"""

    # old
    data_cfg: str = "data_config.json"
    train_cfg: str = "train_config.json"
    model_checkpt: Callable[[int], str] = lambda iteration: f"model.iter_{iteration}.pt"
    model_final: str = "model.final.pt"

    # keep these
    config_holder: str = "config.json"
    checkpoints: str = "checkpoints"
    log: str = "log.jsonl"
    model_checkpt_zanj: Callable[
        [int], str
    ] = lambda iteration: f"model.iter_{iteration}.zanj"
    model_final_zanj: str = "model.final.zanj"
    model_run_dir: Callable[
        [ConfigHolder], str
    ] = (
        lambda cfg: f"{sanitize_fname(cfg.name)}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
