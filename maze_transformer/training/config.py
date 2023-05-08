from __future__ import annotations

import json
import typing
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any, Type

import torch
from muutils.dictmagic import kwargs_to_nested_dict
from muutils.json_serialize import (
    JSONitem,
    SerializableDataclass,
    serializable_dataclass,
    serializable_field,
)
from muutils.tensor_utils import TORCH_OPTIMIZERS_MAP  # type: ignore[import]
from muutils.zanj.torchutil import ConfiguredModel, set_config_class
from transformer_lens import HookedTransformer  # type: ignore[import]
from transformer_lens import HookedTransformerConfig
from transformers import PreTrainedTokenizer

from maze_transformer.dataset.maze_dataset import MazeDatasetConfig
from maze_transformer.dataset.maze_dataset_configs import MAZE_DATASET_CONFIGS
from maze_transformer.dataset.tokenizer import HuggingMazeTokenizer


@serializable_dataclass(kw_only=True)
class BaseGPTConfig(SerializableDataclass):
    """
    Add a name property and serialization to HookedTransformerConfig
    """

    name: str
    act_fn: str
    d_model: int
    d_head: int
    n_layers: int

    weight_processing: dict[str, bool] = serializable_field(
        default_factory=lambda: dict(
            are_layernorms_folded=False,
            are_weights_processed=False,
        )
    )


# ==================================================


def _optimizer_save_fn(optim: Type[torch.optim.Optimizer]) -> str:
    """convert torch optimizer to string, while checking that the conversion is reversible"""
    optim_name: str = optim.__name__
    assert optim_name in TORCH_OPTIMIZERS_MAP
    assert TORCH_OPTIMIZERS_MAP[optim_name] == optim
    return optim_name


@serializable_dataclass(kw_only=True)
class TrainConfig(SerializableDataclass):
    """full training configuration"""

    name: str

    optimizer: Type[torch.optim.Optimizer] = serializable_field(  # type: ignore
        default_factory=lambda: torch.optim.RMSprop,
        serialization_fn=_optimizer_save_fn,
        loading_fn=lambda data: TORCH_OPTIMIZERS_MAP[data["optimizer"]],
    )

    optimizer_kwargs: dict[str, Any] = serializable_field(  # type: ignore
        default_factory=lambda: dict(lr=0.000001)
    )

    @property
    def get_optimizer(self, params) -> Type[torch.optim.Optimizer]:
        return self.optimizer(params, **self.optimizer_kwargs)

    batch_size: int = serializable_field(default=128)

    dataloader_cfg: dict = serializable_field(  # type: ignore
        default_factory=lambda: dict(
            shuffle=True,
            num_workers=16,  # make this smaller if you're not running on a big cluster probably
            persistent_workers=True,
            drop_last=True,
            # collate_fn = None, # we pad the tensors in the Dataset object
            # batch_size = None, # see batchsize in the encompassing TrainConfig
        )
    )

    print_loss_interval: int = serializable_field(default=1000)
    checkpoint_interval: int = serializable_field(default=50000)


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
        name="tuned-v1",
        act_fn="gelu",
        d_model=384,
        d_head=64,
        n_layers=6,
    ),
    BaseGPTConfig(
        name="gpt2-small",
        act_fn="gelu",
        d_model=384,  # half of gpt2-small
        d_head=64,  # match gpt-2 small
        n_layers=12,  # half of gpt2-small
    ),
    # this one is just for integration tests
    BaseGPTConfig(
        name="nano-v1",
        act_fn="gelu",
        d_model=8,
        d_head=4,
        n_layers=2,
    ),
]

GPT_CONFIGS: dict[str, BaseGPTConfig] = {cfg.name: cfg for cfg in _GPT_CONFIGS_LIST}

_TRAINING_CONFIG_LIST: list[TrainConfig] = [
    TrainConfig(
        name="test-v1",
        optimizer=torch.optim.RMSprop,
        optimizer_kwargs=dict(lr=0.0001),
        batch_size=16,
        dataloader_cfg=dict(
            shuffle=True,
            num_workers=0,
            drop_last=False,
        ),
        print_loss_interval=100,
        checkpoint_interval=1000,
    ),
    TrainConfig(
        name="tiny-v1",
        optimizer=torch.optim.RMSprop,
        optimizer_kwargs=dict(lr=0.000001),
        batch_size=32,
        dataloader_cfg=dict(
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            drop_last=True,
        ),
        print_loss_interval=1000,
        checkpoint_interval=5000,
    ),
    TrainConfig(
        name="gpt2-small",
        optimizer=torch.optim.AdamW,
        optimizer_kwargs=dict(lr=6e-4, weight_decay=1e-1, betas=(0.9, 0.95)),
        batch_size=64,
        dataloader_cfg=dict(
            shuffle=True,
            num_workers=16,
            persistent_workers=True,
            drop_last=True,
        ),
        print_loss_interval=50,
        checkpoint_interval=10000,
    ),
    TrainConfig(
        name="sweep-v1",
        optimizer=torch.optim.AdamW,
        optimizer_kwargs=dict(lr=0.0001),
        batch_size=64,
        dataloader_cfg=dict(
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            drop_last=True,
        ),
        print_loss_interval=1000,
        checkpoint_interval=5000,
    ),
]


TRAINING_CONFIGS: dict[str, TrainConfig] = {
    cfg.name: cfg for cfg in _TRAINING_CONFIG_LIST
}


@serializable_dataclass(kw_only=True)
class ConfigHolder(SerializableDataclass):
    """
    Handles any logic that moves data between the configs below it.
    """

    dataset_cfg: MazeDatasetConfig
    model_cfg: BaseGPTConfig
    train_cfg: TrainConfig
    name: str = serializable_field(default="default")
    pretrainedtokenizer_kwargs: dict[str, JSONitem] | None = serializable_field(
        default=None
    )

    @property
    def seed(self) -> int:
        return self.dataset_cfg.seed

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        """if pretrained tokenizer kwargs are provided, use those, otherwise use the HuggingMazeTokenizer derived from the dataset_cfg"""
        if self.pretrainedtokenizer_kwargs is not None:
            return PreTrainedTokenizer(**self.pretrainedtokenizer_kwargs)
        else:
            return HuggingMazeTokenizer(self.dataset_cfg)

    @cached_property
    def hooked_transformer_cfg(self) -> HookedTransformerConfig:
        return HookedTransformerConfig(
            act_fn=self.model_cfg.act_fn,
            d_model=self.model_cfg.d_model,
            d_head=self.model_cfg.d_head,
            n_layers=self.model_cfg.n_layers,
            n_ctx=self.dataset_cfg.seq_len_max,
            d_vocab=len(self.dataset_cfg.token_arr),
        )

    def transformer_config(self) -> HookedTransformerConfig:
        warnings.warn(
            "cfg.transformer_config() is deprecated, use cfg.hooked_transformer_cfg or cfg.create_model_zanj() instead",
            DeprecationWarning,
        )
        return self.hooked_transformer_cfg

    def create_model(self) -> HookedTransformer:
        return HookedTransformer(
            cfg=self.hooked_transformer_cfg,
            tokenizer=self.tokenizer,
        )

    def create_model_zanj(self) -> ZanjHookedTransformer:
        return ZanjHookedTransformer(self)

    @classmethod
    def get_config_multisource(
        cls,
        cfg: ConfigHolder | None = None,
        cfg_file: str | Path | None = None,
        cfg_names: typing.Sequence[str] | None = None,
        kwargs_in: dict | None = None,
    ) -> ConfigHolder:
        """pass one of cfg object, file, or list of names. Any kwargs will be applied to the config object (and should start with 'cfg.')
        
        cfg_names should be either `(dataset_cfg_name,model_cfg_name,train_cfg_name)` or the same with collective name at the end

        valid name keys:
            - dataset_cfg_name: {dataset_cfg_names}
            - model_cfg_name: {model_cfg_names}
            - train_cfg_name: {train_cfg_names}
        """.format(
            dataset_cfg_names=str(list(MAZE_DATASET_CONFIGS.keys())),
            model_cfg_names=str(list(GPT_CONFIGS.keys())),
            train_cfg_names=str(list(TRAINING_CONFIGS.keys())),
        )

        config: ConfigHolder
        assert (
            sum(1 for x in (cfg, cfg_file, cfg_names) if x is not None) == 1
        ), "Must provide exactly one of cfg, cfg_file, or cfg_names"

        if cfg is not None:
            assert cfg_names is None, "Must provide either cfg or cfg_names"
            config = cfg
        elif cfg_file is not None:
            with open(cfg_file) as f:
                config = ConfigHolder.load(json.load(f))
        elif cfg_names is not None:
            assert (
                len(cfg_names) == 3 or len(cfg_names) == 4
            ), "cfg_names must be (dataset_cfg_name,model_cfg_name,train_cfg_name) or the same with collective name at the end"
            dataset_cfg_name: str
            model_cfg_name: str
            train_cfg_name: str
            name: str
            if len(cfg_names) == 3:
                dataset_cfg_name, model_cfg_name, train_cfg_name = cfg_names
                name = f"multsrc_{dataset_cfg_name}_{model_cfg_name}_{train_cfg_name}"
            else:
                dataset_cfg_name, model_cfg_name, train_cfg_name, name = cfg_names
            config = ConfigHolder(
                name=name,
                dataset_cfg=MAZE_DATASET_CONFIGS[dataset_cfg_name],
                model_cfg=GPT_CONFIGS[model_cfg_name],
                train_cfg=TRAINING_CONFIGS[train_cfg_name],
            )

        else:
            raise ValueError(
                "Must provide exactly one of cfg, cfg_file, or cfg_names. this state should be unreachable btw."
            )

        # update config with kwargs
        if kwargs_in:
            kwargs_dict: dict = kwargs_to_nested_dict(
                kwargs_in, sep=".", strip_prefix="cfg.", when_unknown_prefix="raise"
            )
            config.update_from_nested_dict(kwargs_dict)

        return config


@set_config_class(ConfigHolder)
class ZanjHookedTransformer(ConfiguredModel, HookedTransformer):
    """A hooked transformer that is configured by a ConfigHolder

    the inheritance order is critical here -- super() does not call parent, but calls the next class in the MRO
    So, we need ConfiguredModel to take the ConfigHolder and pass kwargs to HookedTransformer
    """

    def __init__(self, cfg_holder: ConfigHolder) -> None:
        super().__init__(
            # for ConfiguredModel
            zanj_model_config=cfg_holder,
            # for HookedTransformer
            cfg=cfg_holder.hooked_transformer_cfg,
            tokenizer=cfg_holder.tokenizer,
        )

    def _load_state_dict_wrapper(
        self,
        state_dict: dict[str, Any],
        **kwargs,
    ) -> None:
        """this is a wrapper around the _load_state_dict function that allows us to do extra things when loading a state dict

        # kwargs:
        - `recover_exact = False` disables `center_writing_weights` and `center_unembed` if set to true
        - `fold_ln = False` folds the layernorms if set to true
        - `refactor_factored_attn_matrices = False` refactors the factored attention matrices if set to true, this might cause accuracy issues according to @valedan

        """

        recover_exact: bool = kwargs.get("recover_exact", False)
        fold_ln: bool = kwargs.get("fold_ln", False)
        refactor_factored_attn_matrices: bool = kwargs.get(
            "refactor_factored_attn_matrices", False
        )

        if (
            self.zanj_model_config.model_cfg.weight_processing["are_layernorms_folded"]
            and fold_ln
        ):
            raise ValueError(
                f"Cannot fold layernorms twice! the saved model already has layernorms folded\n{kwargs = }"
            )

        if recover_exact and (fold_ln or refactor_factored_attn_matrices):
            raise ValueError(
                "Can't recover exact weights if the layernorm is to be folded, or the attention matrices are to be refactored\n{kwargs = }"
            )

        self.zanj_model_config.model_cfg.weight_processing["are_layernorms_folded"] = (
            self.zanj_model_config.model_cfg.weight_processing["are_layernorms_folded"]
            or fold_ln
        )
        self.zanj_model_config.model_cfg.weight_processing[
            "are_weights_processed"
        ] = self.zanj_model_config.model_cfg.weight_processing[
            "are_weights_processed"
        ] or (
            not recover_exact
        )

        self.load_and_process_state_dict(
            state_dict,
            fold_ln=False,
            center_writing_weights=not recover_exact,
            center_unembed=not recover_exact,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
        )
        # We're folding layernorm, but not using HookedTransformer.from_pretrained
        # This means when torch.load_state_dict is invoked by transformer_lens, it
        # will complain about the fact that we deleted layernorm from the state_dict
        # NOTE temporary fix until https://github.com/neelnanda-io/TransformerLens/issues/219 is resolved

        self.process_weights_(
            fold_ln=fold_ln,
            center_writing_weights=not recover_exact,
            center_unembed=not recover_exact,
            refactor_factored_attn_matrices=False,
            move_state_dict_to_device=not recover_exact,
        )
        self.setup()  # Re-attach layernorm hooks by calling setup
        self.eval()
