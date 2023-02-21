import json
from pathlib import Path

import torch
from muutils.logger import Logger
from transformers import PreTrainedTokenizer

from maze_transformer.generation.latticemaze import SPECIAL_TOKENS
from maze_transformer.training.config import (GPT_CONFIGS, TRAINING_CONFIGS,
                                              ConfigHolder)
from maze_transformer.training.mazedataset import MazeDataset
from maze_transformer.training.training import (TRAIN_SAVE_FILES,
                                                get_dataloader, setup_logger,
                                                train)


def train_model(basepath: str, cfg_name: str = "tiny-v1"):
    dataset = MazeDataset.disk_load(basepath, do_config=True, do_tokenized=True)

    tokenizer = PreTrainedTokenizer(
        bos_token=SPECIAL_TOKENS["padding"],
        eos_token=SPECIAL_TOKENS["padding"],
        pad_token=SPECIAL_TOKENS["padding"],
    )

    # TODO: separate names for training and model config
    cfg: ConfigHolder = ConfigHolder(
        dataset_cfg=dataset.cfg,
        model_cfg=GPT_CONFIGS[cfg_name],
        train_cfg=TRAINING_CONFIGS[cfg_name],
        tokenizer=tokenizer
    )

    with open(Path(basepath) / TRAIN_SAVE_FILES.config_holder, "w") as f:
        json.dump(cfg.serialize(), f, indent="\t")

    output_dir_name = TRAIN_SAVE_FILES.train_dir_format(
        cfg.dataset_cfg, cfg.train_cfg
    )
    output_path: Path = Path(basepath) / output_dir_name
    (output_path / TRAIN_SAVE_FILES.checkpoints).mkdir(parents=True)


    logger: Logger = setup_logger(
        output_path=output_path,
        config=cfg,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader = get_dataloader(dataset, cfg, logger)

    train(dataloader, cfg, logger, output_path, device)


if __name__ == "__main__":
    import fire

    fire.Fire(train_model)
