from typing import Any, Dict

import pytest
import torch

from maze_transformer.training.config import TrainConfig


def test_serialize_and_load_default_values():
    config = TrainConfig(name="test")
    serialized = config.serialize()
    loaded = TrainConfig.load(serialized)
    assert loaded == config


def test_serialize_custom_values():
    serialized = _custom_train_config().serialize()
    assert serialized["optimizer"] == "SGD"
    assert serialized == _custom_serialized_config()


def test_load_custom_values():
    loaded = TrainConfig.load(_custom_serialized_config())
    assert loaded.optimizer == torch.optim.SGD
    assert loaded.diff(_custom_train_config()) == {}
    assert loaded == _custom_train_config()


def _custom_train_config() -> TrainConfig:
    return TrainConfig(
        name="test",
        optimizer=torch.optim.SGD,
        optimizer_kwargs=dict(lr=0.01, momentum=0.9),
        batch_size=64,
        dataloader_cfg=dict(num_workers=8, drop_last=False),
        intervals=dict(
            print_loss=100,
            checkpoint=10,
            eval_fast=20,
            eval_slow=10,
        ),
        evals_max_new_tokens=16,
        validation_dataset_cfg=100,
    )


def _custom_serialized_config() -> Dict[Any, Any]:
    return {
        "name": "test",
        "optimizer": "SGD",
        "optimizer_kwargs": {"lr": 0.01, "momentum": 0.9},
        "batch_size": 64,
        "dataloader_cfg": {"num_workers": 8, "drop_last": False},
        "intervals": {
            "print_loss": 100,
            "checkpoint": 10,
            "eval_fast": 20,
            "eval_slow": 10,
        },
        "intervals_count": None,
        "evals_max_new_tokens": 16,
        "validation_dataset_cfg": 100,
        "__format__": "TrainConfig(SerializableDataclass)",
    }


def test_load_invalid_data():
    with pytest.raises(AssertionError):
        TrainConfig.load("not a dictionary")


def test_load_missing_fields():
    loaded = TrainConfig.load({"name": "test"})
    assert loaded == TrainConfig(name="test")


def test_load_extra_field_is_ignored():
    serialized = _custom_serialized_config()
    serialized["extra_field"] = "extra_field_value"
    loaded = TrainConfig.load(serialized)
    assert not hasattr(loaded, "extra_field")
