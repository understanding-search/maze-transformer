
import pytest

from maze_transformer.training.config import GPT_CONFIGS, BaseGPTConfig


def test_serialize_and_load_tiny_v1():
    config = GPT_CONFIGS["tiny-v1"]
    serialized = config.serialize()
    loaded = BaseGPTConfig.load(serialized)
    assert loaded == config

def test_serialize_custom_values():
    serialized = _custom_base_gpt_config().serialize()
    assert serialized == _custom_serialized_config()

def test_load_custom_values():
    loaded = BaseGPTConfig.load(_custom_serialized_config())
    assert loaded == _custom_base_gpt_config()

def test_load_invalid_data():
    with(pytest.raises(TypeError)):
        config = BaseGPTConfig.load("not a dict")

def test_load_missing_fields():
    # TODO
    pass

def test_load_extra_field_is_ignored():
    # TODO
    pass

def _custom_base_gpt_config():
    return BaseGPTConfig(
        name = "test",
        act_fn="relu",
        d_model=1,
        d_head=1,
        n_ctx=1,
        n_layers=1
    )

# TODO: I think we should change this so that we only serialize the values
# set when we instantiate the BaseGPTConfig. Otherwise we end up
# with loads of bloat in the config, making it harder to parse
def _custom_serialized_config():
    return {
        "name": "test",
        "act_fn": "relu",
        "d_model": 1,
        "d_head": 1,
        "n_ctx": 1,
        "n_layers": 1,
        # Additional default values
        'attention_dir': 'causal',
        'attn_only': False,
        'attn_types': None,
        'checkpoint_index': None,
        'checkpoint_label_type': None,
        'checkpoint_value': None,
        'd_mlp': 4,
        'd_vocab': -1,
        'd_vocab_out': -1,
        'device': 'cpu',
        'eps': 1e-05,
        'final_rms': False,
        'from_checkpoint': False,
        'init_mode': 'gpt2',
        'init_weights': True,
        'initializer_range': 0.8,
        'model_name': 'custom',
        'n_heads': 1,
        'n_params': 12,
        'normalization_type': 'LN',
        'original_architecture': None,
        'parallel_attn_mlp': False,
        'positional_embedding_type': 'standard',
        'rotary_dim': None,
        'scale_attn_by_inverse_layer_idx': False,
        'seed': None,
        'tokenizer_name': None,
        'use_attn_result': False,
        'use_attn_scale': True,
        'use_hook_tokens': False,
        'use_local_attn': False,
        'window_size': None
    }