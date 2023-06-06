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
from transformer_lens import HookedTransformer  # type: ignore[import]
from transformer_lens import HookedTransformerConfig
from transformers import PreTrainedTokenizer
from zanj.loading import load_item_recursive
from zanj.torchutil import ConfiguredModel, set_config_class

from maze_transformer.dataset.dataset import GPTDatasetConfig
from maze_transformer.dataset.maze_dataset_configs import MAZE_DATASET_CONFIGS
from maze_transformer.dataset.tokenizer import HuggingMazeTokenizer


@serializable_dataclass(kw_only=True, properties_to_serialize=["n_heads"])
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

    @property
    def n_heads(self) -> int:
        return self.d_model // self.d_head

    def summary(self) -> dict:
        """return a human-readable summary of the config"""
        return dict(
            name=self.name,
            act_fn=self.act_fn,
            d_model=self.d_model,
            d_head=self.d_head,
            n_layers=self.n_layers,
            weight_processing=self.weight_processing,
            n_heads=self.n_heads,
        )


# ==================================================

_DEFAULT_INTERVAL_COUNTS: typing.Callable[[], dict[str, int]] = lambda : dict(
    print_loss=100,
    checkpoint=10,
    eval_fast=20,
    eval_slow=10,
)

def _intervals_loading_fn(data: dict) -> dict[str, int]:
    if "intervals" in data:
        return data["intervals"]
    else:
        warnings.warn("`intervals` not found in config (probably trying to load a legacy config), using None!")
        return None


def _optimizer_save_fn(optim: Type[torch.optim.Optimizer]) -> str:
    """convert torch optimizer to string, while checking that the conversion is reversible"""
    optim_name: str = optim.__name__
    assert optim_name in TORCH_OPTIMIZERS_MAP
    assert TORCH_OPTIMIZERS_MAP[optim_name] == optim
    return optim_name

class ValueWarning(ValueError):
    """raised when a value is not found, but not a fatal error"""
    pass


@serializable_dataclass(kw_only=True)
class TrainConfig(SerializableDataclass):
    """full training configuration

    # Usage:
    - get the optimizer via calling `train_cfg.get_optimizer(model.parameters())`
    - get the intervals in terms of batches via `train_cfg.intervals_batches`
    
    # Parameters

    - `name: str`: name of the training configuration
    - `optimizer: Type[torch.optim.Optimizer]`: optimizer class to use
    - `optimizer_kwargs: dict[str, Any]`: kwargs to pass to the optimizer
    - `batch_size: int`: batch size
    - `dataloader_cfg: dict`: kwargs to pass to the dataloader
    - `intervals: dict[str, int]`: intervals at which to perform certain actions:
        "print_loss", "checkpoint", "eval_fast", "eval_slow"
    - `intervals_count: dict[str, int]`: how many of each action to do over the course of the training run

    """

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

    intervals: dict[str, int]|None = serializable_field(
        default=None,
        loading_fn=_intervals_loading_fn,
    )

    intervals_count: dict[str, int] = serializable_field(
        default=None,
        loading_fn=lambda data: data.get("intervals_count", None),
    )

    def get_intervals(
            self, 
            dataset_n_samples: int|None = None,
            use_defaults_if_missing: bool = True,
            mod_batch_size: bool = True,
        ) -> dict[str, int]:
        """get the intervals"""
        
        # handle the case where both are missing
        if (self.intervals is None) and (self.intervals_count is None):
            if use_defaults_if_missing:
                self.intervals_count = _DEFAULT_INTERVAL_COUNTS()
            else:
                raise ValueError("both `intervals` and `intervals_count` are missing, and `use_defaults_if_missing` is False. Don't know what intervals to use!")

        
        # handle the case where we compute the intervals from counts
        if (self.intervals_count is not None) and (dataset_n_samples is not None):
            intervals_new: dict[str, int] = {
                k: (
                    dataset_n_samples // v
                    if v > 0 
                    else dataset_n_samples + 1 # setting a count to 0 means "dont do it"
                )
                for k, v in self.intervals_count.items()
            }
            
            if self.intervals is not None:
                self.intervals.update(intervals_new)
            else:
                self.intervals = intervals_new

        # checks
        try:
            match (self.intervals is None, self.intervals_count is None, dataset_n_samples is None):
                case (True, True, True):
                    raise ValueError("need some intervals or use defaults")
                case (True, True, False):
                    raise ValueError(f"need some intervals or use defaults")
                case (True, False, True):
                    raise ValueError(f"can't compute intervals from counts without knowing the dataset size!")
                case (True, False, False):
                    raise ValueError(f"this should be inaccessible, since we should have computed the intervals from counts")
                case (False, True, True):
                    # this is fine, we just use the intervals that are already there
                    pass
                case (False, True, False):
                    raise ValueWarning(f"We can't compute the intervals from the counts without knowing the dataset size! However, intervals aren't None so we'll just use that")
                case (False, False, True):
                    raise ValueWarning(f"You gave a dataset size, but no counts. We'll just use the intervals that are already there")
        except (ValueError,ValueWarning) as e:
            _debug_vals: str = f"{dataset_n_samples=}, {use_defaults_if_missing=}, {mod_batch_size=}, {self.intervals=}, {self.intervals_count=}"
            if isinstance(e, ValueWarning):
                warnings.warn(f"{_debug_vals}\ntriggered warning:\n{e}")
            else:
                raise ValueError(f"{_debug_vals}\ntriggered error") from e

        # actually return the intervals
        if mod_batch_size:
            return {
                k: max(1, v // self.batch_size) 
                for k, v in self.intervals.items()
            }
        else:
            return self.intervals


    
    def summary(self) -> dict:
        """return a human-readable summary of the config"""
        return dict(
            name=self.name,
            optimizer=self.optimizer.__name__,
            optimizer_kwargs=self.optimizer_kwargs,
            batch_size=self.batch_size,
            dataloader_cfg=self.dataloader_cfg,
            print_loss_interval=self.print_loss_interval,
            checkpoint_interval=self.checkpoint_interval,
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

    dataset_cfg: GPTDatasetConfig = serializable_field(
        # serialization_fn=lambda self: json_serialize(self.serializ,
        loading_fn=lambda data: load_item_recursive(
            data["dataset_cfg"],
            path=tuple("dataset_cfg"),
        ),
        assert_type=True,
    )
    model_cfg: BaseGPTConfig
    train_cfg: TrainConfig
    name: str = serializable_field(default="default")
    pretrainedtokenizer_kwargs: dict[str, JSONitem] | None = serializable_field(
        default=None
    )

    def summary(self) -> str:
        return {
            "name": self.name,
            "dataset_cfg": self.dataset_cfg.summary(),
            "model_cfg": self.model_cfg.summary(),
            "train_cfg": self.train_cfg.summary(),
            "pretrainedtokenizer_kwargs": self.pretrainedtokenizer_kwargs,
        }

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
            d_vocab=len(self.dataset_cfg.token_arr),  # TODO: get this from tokenizer
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
