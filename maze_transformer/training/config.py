from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Type

import torch
from muutils.json_serialize import (  # type: ignore[import]
    dataclass_loader_factory,
    dataclass_serializer_factory,
)
from muutils.tensor_utils import TORCH_OPTIMIZERS_MAP  # type: ignore[import]
from transformer_lens import HookedTransformer  # type: ignore[import]
from transformer_lens import HookedTransformerConfig
from transformers import PreTrainedTokenizer

from maze_transformer.training.dataset import GPTDatasetConfig
from maze_transformer.training.mazedataset import MazeDatasetConfig


@dataclass(kw_only=True)
class BaseGPTConfig:
    """
    Add a name property and serialization to HookedTransformerConfig
    """

    name: str
    act_fn: str
    d_model: int
    d_head: int
    n_layers: int


BaseGPTConfig.serialize = dataclass_serializer_factory(BaseGPTConfig)
BaseGPTConfig.load = dataclass_loader_factory(BaseGPTConfig)


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

_GPT_CONFIGS_LIST: list[BaseGPTConfig] = [
    BaseGPTConfig(
        name="tiny-v1",
        act_fn="gelu",
        d_model=32,
        d_head=16,
        n_layers=4,
    ),
    BaseGPTConfig(
        name="gpt2-small",
        act_fn="gelu",
        d_model=768,
        d_head=12,
        n_layers=12,
    )
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
    ),
    TrainConfig(
        name="tiny-v1-long",
        optimizer=torch.optim.RMSprop,
        optimizer_kwargs=dict(lr=0.000001),
        batch_size=64,
        dataloader_cfg=dict(
            shuffle=True,
            num_workers=16,
            persistent_workers=True,
            drop_last=True,
        ),
        print_loss_interval=1000,
        checkpoint_interval=5000,
        epochs=20,
    ),
    TrainConfig(
        name="gpt2-small",
        optimizer=torch.optim.AdamW,
        optimizer_kwargs=dict(lr=0.00001, weight_decay=0.01),
        batch_size=64,
        dataloader_cfg=dict(
            shuffle=True,
            num_workers=16,
            persistent_workers=True,
            drop_last=True,
        ),
        print_loss_interval=5000,
        checkpoint_interval=50000,
        # epochs=1,
    )
]


TRAINING_CONFIGS: dict[str, TrainConfig] = {
    cfg.name: cfg for cfg in _TRAINING_CONFIG_LIST
}


@dataclass
class ConfigHolder:
    """
    Handles any logic that moves data between the configs below it.
    """

    train_cfg: TrainConfig
    dataset_cfg: GPTDatasetConfig | MazeDatasetConfig
    model_cfg: BaseGPTConfig
    tokenizer: PreTrainedTokenizer | None

    def create_model(self) -> HookedTransformer:
        hooked_transformer_cfg = HookedTransformerConfig(
            act_fn=self.model_cfg.act_fn,
            d_model=self.model_cfg.d_model,
            d_head=self.model_cfg.d_head,
            n_layers=self.model_cfg.n_layers,
            n_ctx=self.dataset_cfg.seq_len_max,
            d_vocab=len(self.dataset_cfg.token_arr),
        )

        return HookedTransformer(cfg=hooked_transformer_cfg, tokenizer=self.tokenizer)

    def serialize(self):
        return dict(
            train_cfg=self.train_cfg.serialize(),
            dataset_cfg=self.dataset_cfg.serialize(),
            model_cfg=self.model_cfg.serialize(),
        )

    def __repr__(self) -> str:
        return str(self.serialize())

    @classmethod
    def load(cls, serialized: Dict[str, Dict[Any, Any]]):
        return cls(
            train_cfg=TrainConfig.load(serialized["train_cfg"]),
            dataset_cfg=MazeDatasetConfig.load(serialized["dataset_cfg"]),
            model_cfg=BaseGPTConfig.load(serialized["model_cfg"]),
            tokenizer=None,
        )
