from __future__ import annotations
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable

import torch
from muutils.json_serialize import (
    dataclass_loader_factory,
    dataclass_serializer_factory,
)
from muutils.tensor_utils import DTYPE_MAP, TORCH_OPTIMIZERS_MAP
from transformers import OpenAIGPTConfig

from maze_transformer.training.dataset import GPTDatasetConfig
from transformer_lens import HookedTransformerConfig

DEVICE_OVERRIDE: torch.device | None = (
    torch.device("cuda:0") if torch.cuda.is_available() else None
)

TokenizerFunction = Callable[[list[str]], list[int]]


# ==================================================


@dataclass(frozen=True, kw_only=True)
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
    optimizer: torch.optim.Optimizer = torch.optim.RMSprop
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


TrainConfig.serialize = dataclass_serializer_factory(
    TrainConfig,
    special_serializers=dict(
        _optimizer_name=lambda self: self.optimizer.__name__,
    ),
    fields_exclude=["optimizer"],
)

TrainConfig.load = dataclass_loader_factory(
    TrainConfig,
    special_loaders=dict(
        optimizer=lambda d: TORCH_OPTIMIZERS_MAP[d["_optimizer_name"]],
    ),
)

# actual configuration setups
# ==================================================

# TODO: modify existing configurations (base and training) below to work with the updated classes.
# write tests: 
    # - loading and serialization of TrainConfig, BaseGPTConfig, TopLevelConfig
    # - make sure we can instantiate HookedTransformer using BaseGPTConfig
    # - get `train`, `eval_model`, `plot_attention`` to run`
    # - run train function for a single gradient update (sanity check) - make sure it doesn't crash


_GPT_CONFIGS_LIST: list[BaseGPTConfig] = [
    BaseGPTConfig(
        name="tiny-v1",
        d_model=32,
        n_layers=4,
        n_heads=2,
    ),
    BaseGPTConfig(
        name="medium-v1",
        d_model=128,
        n_layers=8,
        n_heads=4,
    ),
]

GPT_CONFIGS: dict[str, BaseGPTConfig] = {cfg.name: cfg for cfg in _GPT_CONFIGS_LIST}

_TRAINING_CONFIG_LIST: list[TrainConfig] = [
    TrainConfig(
        name="tiny-v1",
        base_gpt_cfg=GPT_CONFIGS["tiny-v1"],
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
        seq_len_max=90,
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
