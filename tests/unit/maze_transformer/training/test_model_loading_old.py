"""
test loading of old style models
"""
import json
from pathlib import Path

import pytest
import torch

from maze_transformer.evaluation.eval_model import load_model_with_configs
from maze_transformer.training.config import ConfigHolder, TrainConfig, MazeDatasetConfig, BaseGPTConfig
from maze_transformer.training.wandb_logger import WandbProject
from scripts.create_dataset import create_dataset
from scripts.train_model import train_model


@pytest.mark.usefixtures("temp_dir")
def test_model_loading_notrain(temp_dir):
    cfgholder: ConfigHolder = ConfigHolder(
        train_cfg=TrainConfig(name="test_cfg_save-train"),
        dataset_cfg=MazeDatasetConfig(name="test_cfg_save-data", grid_n=5, n_mazes=10),
        model_cfg=BaseGPTConfig(
            name="test_cfg_save-model",
            act_fn="relu", # need an actual function here since HookedTransformer will complain otherwise
            d_model=128,
            d_head=64,
            n_layers=4,
        ),
    )

    model_orig = cfgholder.create_model()
    torch.save(model_orig.state_dict(), temp_dir / "model.pt")
    with open(temp_dir / "config.json", "w") as f:
        json.dump(cfgholder.serialize(), f)

    # Load model manually without folding
    with open(temp_dir / "config.json", "r") as f:
        cfgholder_loaded = ConfigHolder.load(json.load(f))
    model_state_dict = torch.load(temp_dir / "model.pt")
    model_loaded = cfgholder_loaded.create_model()
    model_loaded.load_state_dict(model_state_dict)

    # Random input tokens
    input_sequence = torch.randint(
        low=0,
        high=len(cfgholder.dataset_cfg.token_arr),
        size=(1, min(cfgholder.dataset_cfg.seq_len_max, 10)),
    )

    # Check for equality in argsort (absolute values won't be equal due to centering the unembedding weight matrix)
    # Alternatively could apply normalization (e.g. softmax) and check with atol v-small
    # (roughly 1E-7 for float error on logexp I think)
    assert torch.all(
        model_orig(input_sequence.clone()).argsort()
        == model_loaded(input_sequence.clone()).argsort()
    )
