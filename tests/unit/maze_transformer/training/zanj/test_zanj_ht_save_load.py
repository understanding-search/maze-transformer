from pathlib import Path

import torch
from jaxtyping import Int
from muutils.zanj import ZANJ
from muutils.zanj.torchutil import (
    ConfigMismatchException,
    assert_model_cfg_equality,
    assert_model_exact_equality,
)

from maze_transformer.training.config import (
    BaseGPTConfig,
    ConfigHolder,
    TrainConfig,
    ZanjHookedTransformer,
)
from maze_transformer.dataset.maze_dataset import MazeDatasetConfig

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


def _check_except_config_equality_modulo_weight_processing(
    diff: dict, model_cfg_keys_allowed_diff: list[str]
) -> bool:
    """given the diff between two configs, return True if the only difference is the specified keys under model_cfg"""
    return all(
        [
            set(diff.keys()) == {"model_cfg"},
            set(diff["model_cfg"].keys()) == {"weight_processing"},
            set(diff["model_cfg"]["weight_processing"]["self"].keys())
            == diff["model_cfg"]["weight_processing"]["other"].keys(),
            all(
                k in model_cfg_keys_allowed_diff
                for k in diff["model_cfg"]["weight_processing"]["self"].keys()
            ),
        ]
    )


def _assert_model_output_equality(
    model_a: ZanjHookedTransformer,
    model_b: ZanjHookedTransformer,
    test_sequence_length: int = 10,
    output_atol: float = 1e-7,
):
    """checks that configs are equal (modulo weight processing) and that the models output the same thing"""
    try:
        assert_model_cfg_equality(model_a, model_b)
    except ConfigMismatchException as e:
        if _check_except_config_equality_modulo_weight_processing(
            e.diff, ["are_weights_processed", "are_layernorms_folded"]
        ):
            pass
        else:
            raise e

    # Random input tokens
    dataset_cfg: BaseGPTConfig = model_a.zanj_model_config.dataset_cfg
    input_sequence: Int[torch.Tensor, "1 test_sequence_length"] = torch.randint(
        low=0,
        high=len(dataset_cfg.token_arr),
        size=(1, min(dataset_cfg.seq_len_max, test_sequence_length)),
    )

    # (copied from `test_eval_model.py`)
    # Check for equality in argsort (absolute values won't be equal due to centering the unembedding weight matrix)
    assert torch.all(
        model_a(input_sequence.clone()).argsort()
        == model_b(input_sequence.clone()).argsort()
    )
    # apply normalization (e.g. softmax) and check with atol v-small
    # (roughly 1E-7 for float error on logexp I think)
    output_a: torch.Tensor = torch.nn.functional.softmax(
        model_a(input_sequence.clone()), dim=-1
    )
    output_b: torch.Tensor = torch.nn.functional.softmax(
        model_b(input_sequence.clone()), dim=-1
    )

    assert torch.allclose(output_a, output_b, atol=output_atol)


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
    zanj.save(MODEL, fname)
    model_load = zanj.read(fname)

    assert_model_exact_equality(MODEL, model_load)


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
