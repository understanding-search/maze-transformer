import json
from pathlib import Path

import torch

from muutils.zanj import ZANJ

from maze_transformer.training.config import (
    BaseGPTConfig,
    ConfigHolder,
    MazeDatasetConfig,
    TrainConfig,
    ZanjHookedTransformer,
)


config: ConfigHolder = ConfigHolder(
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

model: ZanjHookedTransformer = ZanjHookedTransformer(config)
model_c: ZanjHookedTransformer = config.create_model_zanj()

def test_configs_correct():
    assert model.config == config
    assert model.cfg == config.hooked_transformer_cfg

    assert model_c.config == config
    assert model_c.cfg == config.hooked_transformer_cfg


def test_model_save():
    fname: Path = Path("tests/_temp/test_model_save.zanj")
    fname.parent.mkdir(parents=True, exist_ok=True)

    zanj: ZANJ = ZANJ()

    zanj.save(model, fname)

    assert fname.exists()

    model_load = zanj.read(fname)

    assert isinstance(model_load, ZanjHookedTransformer)
    assert isinstance(model_load.config, ConfigHolder)
    assert model_load.config == config
    # check state dicts
    assert model.state_dict().keys() == model_load.state_dict().keys()
    keys_failed: list[str] = list()
    for k, v in model.state_dict().items():
        v_load = model_load.state_dict()[k]
        if not (v == v_load).all():
        # if not torch.allclose(v, v_load):
            keys_failed.append(k)
            print(f"failed {k}")
        else:
            print(f"passed {k}")
    assert len(keys_failed) == 0, f"{len(keys_failed)} / {len(model.state_dict())} state dict elements don't match: {keys_failed}"

