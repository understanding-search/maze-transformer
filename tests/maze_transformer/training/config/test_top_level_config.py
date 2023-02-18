import torch

from maze_transformer.training.config import (
    GPT_CONFIGS,
    TRAINING_CONFIGS,
    TopLevelConfig,
)
from maze_transformer.training.mazedataset import MazeDatasetConfig


def test_create_model_forward():
    cfg = _create_top_level_config()
    model = cfg.create_model()

    d_vocab = cfg.dataset_cfg.n_tokens

    tokens = torch.randint(0, d_vocab, (5,))
    logits = model(tokens)

    assert logits.shape == (1, 5, d_vocab)


def test_model_config_has_correct_values():
    cfg = _create_top_level_config()

    model = cfg.create_model()

    assert model.cfg.act_fn == "gelu"
    assert model.cfg.d_model == 32
    assert model.cfg.d_head == 16
    assert model.cfg.n_layers == 4
    assert model.cfg.n_ctx == 512
    assert model.cfg.d_vocab == 25


def test_serialize_and_load():
    cfg = _create_top_level_config()
    serialized = cfg.serialize()
    loaded = TopLevelConfig.load(serialized)

    assert loaded == cfg


def _create_top_level_config() -> TopLevelConfig:
    # It might be better to use custom values in the tests rather than
    # "tiny-v1". I'm not sure.
    train_cfg = TRAINING_CONFIGS["tiny-v1"]
    model_cfg = GPT_CONFIGS["tiny-v1"]
    dataset_cfg = MazeDatasetConfig(name="test", grid_n=4, n_mazes=10)

    return TopLevelConfig(
        train_cfg=train_cfg, model_cfg=model_cfg, dataset_cfg=dataset_cfg
    )
