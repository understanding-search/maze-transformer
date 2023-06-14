import warnings
from pathlib import Path

import torch
from jaxtyping import Int
from zanj import ZANJ
from zanj.torchutil import (
    ConfigMismatchException,
    assert_model_cfg_equality,
    assert_model_exact_equality,
)

from maze_dataset import MazeDatasetConfig
from maze_transformer.training.config import (
    BaseGPTConfig,
    ConfigHolder,
    TrainConfig,
    ZanjHookedTransformer,
)
from maze_transformer.test_helpers.assertions import _assert_model_output_equality

ZANJ_MODEL_CFG: ConfigHolder = ConfigHolder(
    train_cfg=TrainConfig(name="test_cfg_save-train"),
    dataset_cfg=MazeDatasetConfig(name="test_cfg_save-data", grid_n=5, n_mazes=10),
    model_cfg=BaseGPTConfig(
        name="test_cfg_save-model",
        act_fn="relu",  # need an actual function here since HookedTransformer will complain otherwise
        d_model=128,
        d_head=64,
        n_layers=4,
    ),
)

MODEL: ZanjHookedTransformer = ZanjHookedTransformer(ZANJ_MODEL_CFG)
MODEL_FROM_CFG_CREATE: ZanjHookedTransformer = ZANJ_MODEL_CFG.create_model_zanj()





def test_configs_setup_correct():
    assert MODEL.zanj_model_config == ZANJ_MODEL_CFG
    assert MODEL.cfg == ZANJ_MODEL_CFG.hooked_transformer_cfg

    assert MODEL_FROM_CFG_CREATE.zanj_model_config == ZANJ_MODEL_CFG
    assert MODEL_FROM_CFG_CREATE.cfg == ZANJ_MODEL_CFG.hooked_transformer_cfg


def test_model_save_exact():
    fname: Path = Path("tests/_temp/test_model_save_nofold.zanj")
    fname.parent.mkdir(parents=True, exist_ok=True)
    zanj: ZANJ = ZANJ(
        custom_settings={"_load_state_dict_wrapper": {"recover_exact": True}},
    )
    print(f"{MODEL.zanj_model_config.dataset_cfg = }")
    zanj.save(MODEL, fname)
    model_load = zanj.read(fname)

    assert_model_exact_equality(MODEL, model_load)
    _assert_model_output_equality(MODEL, model_load)


def test_model_save_fold_ln():
    fname: Path = Path("tests/_temp/test_model_save.zanj")
    fname.parent.mkdir(parents=True, exist_ok=True)
    zanj: ZANJ = ZANJ(
        custom_settings={"_load_state_dict_wrapper": {"fold_ln": True}},
    )
    zanj.save(MODEL, fname)
    model_load = zanj.read(fname)

    _assert_model_output_equality(MODEL, model_load)


def test_model_save_refactored_attn_matrices():
    fname: Path = Path("tests/_temp/test_model_save.zanj")
    fname.parent.mkdir(parents=True, exist_ok=True)
    zanj: ZANJ = ZANJ(
        custom_settings={
            "_load_state_dict_wrapper": {
                "fold_ln": True,
                "refactor_factored_attn_matrices": True,
            }
        },
    )
    zanj.save(MODEL, fname)
    model_load = zanj.read(fname)

    _assert_model_output_equality(MODEL, model_load)
