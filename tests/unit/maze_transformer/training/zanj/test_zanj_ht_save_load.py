import json
from pathlib import Path

import torch

from muutils.zanj import ZANJ
from muutils.zanj.torchutil import assert_model_cfg_equality, assert_model_exact_equality

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
        act_fn="relu", # need an actual function here since HookedTransformer will complain otherwise
        d_model=128,
        d_head=64,
        n_layers=4,
    ),
)

MODEL: ZanjHookedTransformer = ZanjHookedTransformer(ZANJ_MODEL_CFG)
MODEL_C: ZanjHookedTransformer = ZANJ_MODEL_CFG.create_model_zanj()


def _assert_model_output_equality(model_a: ZanjHookedTransformer, model_b: ZanjHookedTransformer):
    
    # TODO: this is fragile, but I don't know how to do it better
    try:
        assert_model_cfg_equality(model_a, model_b)
    except AssertionError as e:
        if r"configs don't match: {'model_cfg': {'are_weights_processed': {'self': False, 'other': True}}}" in str(e):
            pass
        else:
            raise e

    # Random input tokens
    dataset_cfg = model_a.zanj_model_config.dataset_cfg
    input_sequence = torch.randint(
        low=0,
        high=len(dataset_cfg.token_arr),
        size=(1, min(dataset_cfg.seq_len_max, 10)),
    )

    # (copied from `test_eval_model.py`)
    # Check for equality in argsort (absolute values won't be equal due to centering the unembedding weight matrix)
    assert torch.all(
        model_a(input_sequence.clone()).argsort()
        == model_b(input_sequence.clone()).argsort()
    )
    # apply normalization (e.g. softmax) and check with atol v-small
    # (roughly 1E-7 for float error on logexp I think)
    output_a = torch.nn.functional.softmax(model_a(input_sequence.clone()), dim=-1)
    output_b = torch.nn.functional.softmax(model_b(input_sequence.clone()), dim=-1)

    assert torch.allclose(output_a, output_b, atol=1e-7)

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

    _assert_model_output_equality(MODEL, model_load)
    

def test_model_save_nofold():
    fname: Path = Path("tests/_temp/test_model_save_nofold.zanj")
    fname.parent.mkdir(parents=True, exist_ok=True)
    zanj: ZANJ = ZANJ(
        custom_settings={"_load_state_dict_wrapper": {"recover_exact": True, "fold_ln": False}},
    )
    zanj.save(MODEL, fname)
    model_load = zanj.read(fname)

    assert_model_exact_equality(MODEL, model_load)