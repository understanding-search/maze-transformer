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


zanj_model_config: ConfigHolder = ConfigHolder(
    train_cfg=TrainConfig(name="test_cfg_save-train"),
    dataset_cfg=MazeDatasetConfig(name="test_cfg_save-data", grid_n=5, n_mazes=10),
    model_cfg=BaseGPTConfig(
        name="test_cfg_save-model",
        act_fn="relu", # need an actual function here since HookedTransformer will complain otherwise
        d_model=128,
        d_head=64,
        n_layers=4,
        recover_exact_state_dict=True,
        fold_layernorm=True,
    ),
)

zanj_model_config_nofold: ConfigHolder = ConfigHolder(
    train_cfg=TrainConfig(name="test_cfg_save-train"),
    dataset_cfg=MazeDatasetConfig(name="test_cfg_save-data", grid_n=5, n_mazes=10),
    model_cfg=BaseGPTConfig(
        name="test_cfg_save-model",
        act_fn="relu", # need an actual function here since HookedTransformer will complain otherwise
        d_model=128,
        d_head=64,
        n_layers=4,
        recover_exact_state_dict=True,
        fold_layernorm=False,
    ),
)

model: ZanjHookedTransformer = ZanjHookedTransformer(zanj_model_config)
model_c: ZanjHookedTransformer = zanj_model_config.create_model_zanj()
model_nofold: ZanjHookedTransformer = ZanjHookedTransformer(zanj_model_config_nofold)
model_nofold_c: ZanjHookedTransformer = zanj_model_config_nofold.create_model_zanj()

def test_configs_correct():
    assert model.zanj_model_config == zanj_model_config
    assert model.cfg == zanj_model_config.hooked_transformer_cfg

    assert model_c.zanj_model_config == zanj_model_config
    assert model_c.cfg == zanj_model_config.hooked_transformer_cfg

    assert model_nofold.zanj_model_config == zanj_model_config_nofold
    assert model_nofold.cfg == zanj_model_config_nofold.hooked_transformer_cfg


def test_model_save():
    fname: Path = Path("tests/_temp/test_model_save.zanj")
    fname.parent.mkdir(parents=True, exist_ok=True)

    zanj: ZANJ = ZANJ()

    zanj.save(model, fname)

    assert fname.exists()

    model_load = zanj.read(fname)

    assert isinstance(model_load, ZanjHookedTransformer)
    assert isinstance(model_load.zanj_model_config, ConfigHolder)
    assert model_load.zanj_model_config == zanj_model_config
    # check state dicts
    model_sd_keys: set[str] = set(model.state_dict().keys())
    model_loaded_sd_keys: set[str] = set(model_load.state_dict().keys())
    assert model_sd_keys == model_loaded_sd_keys, f"state dict keys don't match: {model_sd_keys - model_loaded_sd_keys} / {model_loaded_sd_keys - model_sd_keys}"
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

def test_model_save_nofold():
    fname: Path = Path("tests/_temp/test_model_nofold_save.zanj")
    fname.parent.mkdir(parents=True, exist_ok=True)

    zanj: ZANJ = ZANJ()

    zanj.save(model_nofold, fname)

    assert fname.exists()

    model_nofold_load = zanj.read(fname)

    assert isinstance(model_nofold_load, ZanjHookedTransformer)
    assert isinstance(model_nofold_load.zanj_model_config, ConfigHolder)
    assert model_nofold_load.zanj_model_config == zanj_model_config_nofold
    # check state dicts
    model_nofold_sd_keys: set[str] = set(model_nofold.state_dict().keys())
    model_nofold_loaded_sd_keys: set[str] = set(model_nofold_load.state_dict().keys())
    assert model_nofold_sd_keys == model_nofold_loaded_sd_keys, f"state dict keys don't match: {model_nofold_sd_keys - model_nofold_loaded_sd_keys} / {model_nofold_loaded_sd_keys - model_nofold_sd_keys}"
    keys_failed: list[str] = list()
    for k, v in model_nofold.state_dict().items():
        v_load = model_nofold_load.state_dict()[k]
        if not (v == v_load).all():
        # if not torch.allclose(v, v_load):
            keys_failed.append(k)
            print(f"failed {k}")
        else:
            print(f"passed {k}")
    assert len(keys_failed) == 0, f"{len(keys_failed)} / {len(model_nofold.state_dict())} state dict elements don't match: {keys_failed}"