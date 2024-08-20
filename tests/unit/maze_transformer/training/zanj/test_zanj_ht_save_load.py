from pathlib import Path

import pytest
from maze_dataset import MazeDatasetConfig
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode
from zanj import ZANJ
from zanj.torchutil import assert_model_exact_equality

from maze_transformer.test_helpers.assertions import assert_model_output_equality
from maze_transformer.training.config import (
    BaseGPTConfig,
    ConfigHolder,
    TrainConfig,
    ZanjHookedTransformer,
)

ZANJ_MODEL_CFGS: list[ConfigHolder] = [
    ConfigHolder(
        name=f"test-{mt_name}",
        train_cfg=TrainConfig(name="test_cfg_save-train"),
        dataset_cfg=MazeDatasetConfig(name="test_cfg_save-data", grid_n=5, n_mazes=10),
        model_cfg=BaseGPTConfig(
            name="test_cfg_save-model",
            act_fn="relu",  # need an actual function here since HookedTransformer will complain otherwise
            d_model=64,
            d_head=16,
            n_layers=4,
        ),
        maze_tokenizer=mt,
    )
    for mt_name, mt in [
        ("None", None),
        (
            "raster",
            MazeTokenizer(
                tokenization_mode=TokenizationMode.AOTP_UT_rasterized, max_grid_size=10
            ),
        ),
        (
            "uniform",
            MazeTokenizer(
                tokenization_mode=TokenizationMode.AOTP_UT_uniform, max_grid_size=10
            ),
        ),
        (
            "indexed",
            MazeTokenizer(
                tokenization_mode=TokenizationMode.AOTP_CTT_indexed, max_grid_size=10
            ),
        ),
    ]
]


MODELS: list[tuple[ConfigHolder, ZanjHookedTransformer]] = [
    (cfg, ZanjHookedTransformer(cfg)) for cfg in ZANJ_MODEL_CFGS
]


@pytest.mark.parametrize("cfg_model", MODELS, ids=lambda x: x[0].name)
def test_configs_setup_correct(cfg_model: tuple[ConfigHolder, ZanjHookedTransformer]):
    cfg: ConfigHolder
    model: ZanjHookedTransformer
    cfg, model = cfg_model
    assert model.zanj_model_config == cfg
    assert model.cfg == cfg.hooked_transformer_cfg


@pytest.mark.parametrize("cfg_model", MODELS, ids=lambda x: x[0].name)
def test_model_save_exact(cfg_model: tuple[ConfigHolder, ZanjHookedTransformer]):
    cfg: ConfigHolder
    model: ZanjHookedTransformer
    cfg, model = cfg_model
    name: str = cfg.name

    fname: Path = Path(f"tests/_temp/test_model_save_nofold-{name}.zanj")
    fname.parent.mkdir(parents=True, exist_ok=True)
    zanj: ZANJ = ZANJ(
        custom_settings={"_load_state_dict_wrapper": {"recover_exact": True}},
    )
    print(f"{model.zanj_model_config.dataset_cfg = }")
    zanj.save(model, fname)
    model_load = zanj.read(fname)

    assert_model_exact_equality(model, model_load)
    assert_model_output_equality(model, model_load)


@pytest.mark.parametrize("cfg_model", MODELS, ids=lambda x: x[0].name)
def test_model_save_fold_ln(cfg_model: tuple[ConfigHolder, ZanjHookedTransformer]):
    cfg: ConfigHolder
    model: ZanjHookedTransformer
    cfg, model = cfg_model
    name: str = cfg.name

    fname: Path = Path(f"tests/_temp/test_model_save-{name}.zanj")
    fname.parent.mkdir(parents=True, exist_ok=True)
    zanj: ZANJ = ZANJ(
        custom_settings={"_load_state_dict_wrapper": {"fold_ln": True}},
    )
    zanj.save(model, fname)
    model_load = zanj.read(fname)

    assert_model_output_equality(model, model_load)


@pytest.mark.parametrize("cfg_model", MODELS, ids=lambda x: x[0].name)
def test_model_save_refactored_attn_matrices(
    cfg_model: tuple[ConfigHolder, ZanjHookedTransformer]
):
    cfg: ConfigHolder
    model: ZanjHookedTransformer
    cfg, model = cfg_model
    name: str = cfg.name

    fname: Path = Path(f"tests/_temp/test_model_save-{name}.zanj")
    fname.parent.mkdir(parents=True, exist_ok=True)
    zanj: ZANJ = ZANJ(
        custom_settings={
            "_load_state_dict_wrapper": {
                "fold_ln": True,
                "refactor_factored_attn_matrices": True,
            }
        },
    )
    zanj.save(model, fname)
    model_load = zanj.read(fname)

    assert_model_output_equality(model, model_load)
