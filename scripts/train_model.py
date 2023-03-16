import json
from pathlib import Path

from muutils.logger import Logger
from transformers import PreTrainedTokenizer

import wandb
from maze_transformer.generation.latticemaze import SPECIAL_TOKENS
from maze_transformer.training.config import GPT_CONFIGS, TRAINING_CONFIGS, ConfigHolder
from maze_transformer.training.mazedataset import MazeDataset
from maze_transformer.training.training import (
    TRAIN_SAVE_FILES,
    get_dataloader,
    setup_logger,
    train,
)
from maze_transformer.utils.utils import get_device


def train_model(
    basepath: str, training_cfg: str = "tiny-v1", model_cfg: str = "tiny-v1", wandb_project: str="wandb-preview"
):
    dataset = MazeDataset.disk_load(basepath, do_config=True, do_tokenized=True)

    tokenizer = PreTrainedTokenizer(
        bos_token=SPECIAL_TOKENS["padding"],
        eos_token=SPECIAL_TOKENS["padding"],
        pad_token=SPECIAL_TOKENS["padding"],
    )

    cfg: ConfigHolder = ConfigHolder(
        dataset_cfg=dataset.cfg,
        model_cfg=GPT_CONFIGS[model_cfg],
        train_cfg=TRAINING_CONFIGS[training_cfg],
        tokenizer=tokenizer,
    )

    run_id = TRAIN_SAVE_FILES.train_dir_format(cfg.dataset_cfg, cfg.train_cfg)
    output_path: Path = Path(basepath) / run_id
    (output_path / TRAIN_SAVE_FILES.checkpoints).mkdir(parents=True)

    logger: Logger = setup_logger(
        output_path=output_path,
        config=cfg,
    )

    with wandb.init(
        project=wandb_project,
        job_type="train-model",
        config=cfg.serialize(),
        id=run_id,
    ):
        # cfg = wandb.config

        cfg_save_path = output_path / TRAIN_SAVE_FILES.config_holder
        with open(cfg_save_path, "w") as f:
            json.dump(cfg.serialize(), f, indent="\t")

        wandb.save(str(cfg_save_path), base_path=str(cfg_save_path.parent))

        dataloader = get_dataloader(dataset, cfg, logger)
        device = get_device()

        train(dataloader, cfg, logger, output_path, device)


if __name__ == "__main__":
    import fire

    fire.Fire(train_model)
