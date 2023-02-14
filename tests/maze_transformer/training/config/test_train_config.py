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
    assert loaded == _custom_train_config()


def _custom_train_config() -> TrainConfig:
    return TrainConfig(
        name="test",
        epochs=10,
        optimizer=torch.optim.SGD,
        optimizer_kwargs=dict(lr=0.01, momentum=0.9),
        batch_size=64,
        dataloader_cfg=dict(num_workers=8, drop_last=False),
        print_loss_interval=500,
        checkpoint_interval=1000,
    )


def _custom_serialized_config() -> Dict[Any, Any]:
    return {
        "name": "test",
        "epochs": 10,
        "optimizer": "SGD",
        "optimizer_kwargs": {"lr": 0.01, "momentum": 0.9},
        "batch_size": 64,
        "dataloader_cfg": {"num_workers": 8, "drop_last": False},
        "print_loss_interval": 500,
        "checkpoint_interval": 1000,
    }


def test_load_invalid_data():
    with (pytest.raises(TypeError)):
        TrainConfig.load("not a dictionary")


# TODO: I think the behaviour shoud be to throw
# or at least warn if there are fields missing from the serialized
# data. The config should be serialized with all its properties.
# We could achieve this by adding a check for
# cls.__dataclass_fields__.keys() == data.keys() to dataclass_loader_factory.
def test_load_missing_fields():
    loaded = TrainConfig.load({"name": "test"})
    assert loaded == TrainConfig(name="test")

def test_load_extra_field_is_ignored():
    serialized = _custom_serialized_config()
    serialized["extra_field"] = "extra_field_value"
    loaded = TrainConfig.load(serialized)
    assert not hasattr(loaded, "extra_field")
