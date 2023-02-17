from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Type

import torch
from muutils.json_serialize import (  # type: ignore[import]
    dataclass_loader_factory,
    dataclass_serializer_factory,
)
from muutils.tensor_utils import TORCH_OPTIMIZERS_MAP  # type: ignore[import]
from transformer_lens import HookedTransformerConfig  # type: ignore[import]

from maze_transformer.training.dataset import GPTDatasetConfig


@dataclass(kw_only=True)
class BaseGPTConfig(HookedTransformerConfig):
    """
    Add a name property and serialization to HookedTransformerConfig
    """

    name: str

    def serialize(self) -> str:
        return self.to_dict()

    @classmethod
    def load(cls, d: dict) -> BaseGPTConfig:
        return cls.from_dict(d)


# ==================================================


@dataclass(kw_only=True)
class TrainConfig:
    """full training configuration"""

    name: str

    epochs: int = 1
    optimizer: Type[torch.optim.Optimizer] = torch.optim.RMSprop
    optimizer_kwargs: dict[str, Any] = field(default_factory=lambda: dict(lr=0.000001))
    batch_size: int = 128

    dataloader_cfg: dict = field(
        default_factory=lambda: dict(
            shuffle=True,
            num_workers=16,  # make this smaller if you're not running on a big cluster probably
            persistent_workers=True,
            drop_last=True,
            # collate_fn = None, # we pad the tensors in the Dataset object
            # batch_size = None, # see batchsize in the encompassing TrainConfig
        )
    )

    print_loss_interval: int = 1000
    checkpoint_interval: int = 50000


TrainConfig.serialize = dataclass_serializer_factory(  # type: ignore[attr-defined]
    TrainConfig,
    special_serializers=dict(
        optimizer=lambda self: self.optimizer.__name__,
    ),
)

TrainConfig.load = dataclass_loader_factory(  # type: ignore[attr-defined]
    TrainConfig,
    special_loaders=dict(
        optimizer=lambda d: TORCH_OPTIMIZERS_MAP[d["optimizer"]],
    ),
)


# actual configuration setups
# ==================================================

# TODO: modify existing configurations (base and training) below to work with the updated classes.
# write tests:
# - get `train`, `eval_model`, `plot_attention`` to run`
# - Implement and test Serialization and Loading for TopLevelConfig
# - run train function for a single gradient update (sanity check) - make sure it doesn't crash


# TODO: michael check these values
_GPT_CONFIGS_LIST: list[BaseGPTConfig] = [
    BaseGPTConfig(
        name="tiny-v1",
        act_fn="gelu",
        d_model=32,
        d_head=16,
        n_ctx=90,
        n_layers=4,
    ),
]

GPT_CONFIGS: dict[str, BaseGPTConfig] = {cfg.name: cfg for cfg in _GPT_CONFIGS_LIST}

_TRAINING_CONFIG_LIST: list[TrainConfig] = [
    TrainConfig(
        name="tiny-v1",
        optimizer=torch.optim.RMSprop,
        optimizer_kwargs=dict(lr=0.000001),
        batch_size=32,
        dataloader_cfg=dict(
            shuffle=True,
            num_workers=16,
            persistent_workers=True,
            drop_last=True,
        ),
        print_loss_interval=1000,
        checkpoint_interval=5000,
    )
]


TRAINING_CONFIGS: dict[str, TrainConfig] = {
    cfg.name: cfg for cfg in _TRAINING_CONFIG_LIST
}


@dataclass
class TopLevelConfig:
    """
    Handles any logic that moves data between the configs below it.
    """

    train_config: TrainConfig | None
    dataset_config: GPTDatasetConfig | None
    model_config: BaseGPTConfig | None


TopLevelConfig.serialize = dataclass_serializer_factory(TopLevelConfig)
TopLevelConfig.load = dataclass_loader_factory(TopLevelConfig)
