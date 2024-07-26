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
    with pytest.raises(AssertionError):
        BaseGPTConfig.load("not a dict")


def test_load_missing_fields():
    serialized = {
        "name": "test",
        "act_fn": "relu",
        "d_model": 1,
        "d_head": 1,
        "n_layers": 1,
        "n_heads": 1,
        "weight_processing": {
            "are_layernorms_folded": False,
            "are_weights_processed": False,
        },
        "__format__": "BaseGPTConfig(SerializableDataclass)",
    }
    config = BaseGPTConfig.load(serialized)
    assert config == _custom_base_gpt_config()


def test_load_extra_field_is_ignored():
    serialized = _custom_serialized_config()
    serialized["extra_field"] = "extra_field_value"

    loaded = BaseGPTConfig.load(serialized)
    assert not hasattr(loaded, "extra_field")


def _custom_base_gpt_config():
    return BaseGPTConfig(name="test", act_fn="relu", d_model=1, d_head=1, n_layers=1)


def _custom_serialized_config():
    return {
        "name": "test",
        "act_fn": "relu",
        "d_model": 1,
        "d_head": 1,
        "n_layers": 1,
        "n_heads": 1,
        "positional_embedding_type": "standard",
        "weight_processing": {
            "are_layernorms_folded": False,
            "are_weights_processed": False,
        },
        "__format__": "BaseGPTConfig(SerializableDataclass)",
    }
