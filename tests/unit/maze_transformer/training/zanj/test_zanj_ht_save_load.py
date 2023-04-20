from pathlib import Path

from muutils.zanj import ZANJ
from muutils.zanj.torchutil import assert_model_exact_equality

from tests.helpers.assertions import assert_model_output_equality
from maze_transformer.training.config import (
    BaseGPTConfig,
    ConfigHolder,
    MazeDatasetConfig,
    TrainConfig,
    ZanjHookedTransformer,
)

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
MODEL_C: ZanjHookedTransformer = ZANJ_MODEL_CFG.create_model_zanj()


def test_configs_setup_correct():
    assert MODEL.zanj_model_config == ZANJ_MODEL_CFG
    assert MODEL.cfg == ZANJ_MODEL_CFG.hooked_transformer_cfg

    assert MODEL_C.zanj_model_config == ZANJ_MODEL_CFG
    assert MODEL_C.cfg == ZANJ_MODEL_CFG.hooked_transformer_cfg


def test_model_save():
    fname: Path = Path("tests/_temp/test_model_save.zanj")
    fname.parent.mkdir(parents=True, exist_ok=True)
    zanj: ZANJ = ZANJ()
    zanj.save(MODEL, fname)
    model_load = zanj.read(fname)

    assert_model_output_equality(MODEL, model_load)


def test_model_save_nofold():
    fname: Path = Path("tests/_temp/test_model_save_nofold.zanj")
    fname.parent.mkdir(parents=True, exist_ok=True)
    zanj: ZANJ = ZANJ(
        custom_settings={
            "_load_state_dict_wrapper": {"recover_exact": True, "fold_ln": False}
        },
    )
    zanj.save(MODEL, fname)
    model_load = zanj.read(fname)

    assert_model_exact_equality(MODEL, model_load)
