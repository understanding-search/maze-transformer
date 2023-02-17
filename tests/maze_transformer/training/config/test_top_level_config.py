import torch

from maze_transformer.training.config import (
    GPT_CONFIGS,
    TRAINING_CONFIGS,
    TopLevelConfig,
)
from maze_transformer.training.mazedataset import MazeDatasetConfig


def test_create_model():
    train_cfg = TRAINING_CONFIGS["tiny-v1"]
    model_cfg = GPT_CONFIGS["tiny-v1"]
    dataset_cfg = MazeDatasetConfig(name="test", grid_n=4, n_mazes=10)

    cfg = TopLevelConfig(
        train_cfg=train_cfg, model_cfg=model_cfg, dataset_cfg=dataset_cfg
    )

    model = cfg.create_model()

    d_vocab = dataset_cfg.n_tokens

    tokens = torch.randint(0, d_vocab, (5,))
    logits = model(tokens)

    assert logits.shape == (1, 5, d_vocab)
