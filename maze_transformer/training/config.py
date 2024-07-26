from __future__ import annotations

import json
import typing
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any, Type

import torch
from maze_dataset.dataset.configs import MAZE_DATASET_CONFIGS
from maze_dataset.dataset.dataset import GPTDatasetConfig
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode
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

from maze_transformer.tokenizer import HuggingMazeTokenizer


# TODO: replace with muutils
def dynamic_docstring(**doc_params):
    def decorator(func):
        if func.__doc__:
            func.__doc__ = func.__doc__.format(**doc_params)
        return func

    return decorator


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
    positional_embedding_type: str = serializable_field(
        default="standard",
        loading_fn=lambda data: data.get("positional_embedding_type", "standard"),
    )

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
            positional_embedding_type=self.positional_embedding_type,
            weight_processing=self.weight_processing,
            n_heads=self.n_heads,
        )


# ==================================================

_DEFAULT_INTERVAL_COUNTS: typing.Callable[[], dict[str, int]] = lambda: dict(
    print_loss=100,
    checkpoint=10,
    eval_fast=20,
    eval_slow=10,
)


def _intervals_loading_fn(data: dict) -> dict[str, int]:
    if "intervals" in data:
        return data["intervals"]
    else:
        warnings.warn(
            "`intervals` not found in config (probably trying to load a legacy config), using None!"
        )
        return None


def _optimizer_save_fn(optim: Type[torch.optim.Optimizer]) -> str:
    """convert torch optimizer to string, while checking that the conversion is reversible"""
    optim_name: str = optim.__name__
    assert optim_name in TORCH_OPTIMIZERS_MAP
    assert TORCH_OPTIMIZERS_MAP[optim_name] == optim
    return optim_name


@serializable_dataclass(kw_only=True)
class TrainConfig(SerializableDataclass):
    """full training configuration

    # Usage:
    - get the optimizer via calling `train_cfg.get_optimizer(model.parameters())`
    - get the intervals via `train_cfg.get_intervals()`

    # Parameters

    - `name: str`: name of the training configuration
    - `optimizer: Type[torch.optim.Optimizer]`: optimizer class to use
    - `optimizer_kwargs: dict[str, Any]`: kwargs to pass to the optimizer
    - `batch_size: int`: batch size
    - `dataloader_cfg: dict`: kwargs to pass to the dataloader
    - `intervals: dict[str, int]`: intervals at which to perform certain actions:
        "print_loss", "checkpoint", "eval_fast", "eval_slow"
    - `intervals_count: dict[str, int]`: how many of each action to do over the course of the training run
    - `evals_max_new_tokens: int`: how many new tokens to generate during evaluation
    - `validation_dataset_cfg: None|int|GPTDatasetConfig`: validation dataset
        - if `None`, evals are disabled
        - if `int`, a dataset of that size is created by sampling from the training dataset using `torch.utils.data.random_split`
        - if `GPTDatasetConfig`, a dataset is created from the specified config TODO: this is not implemented yet

    """

    name: str
    # TODO: loaders specified here only because of legacy models, remove this after some time and models are updated
    evals_max_new_tokens: int = serializable_field(
        default=8,
        loading_fn=lambda data: data.get("evals_max_new_tokens", 8),
    )
    validation_dataset_cfg: None | int | GPTDatasetConfig = serializable_field(
        default=None,
        loading_fn=lambda data: data.get("validation_dataset_cfg", None),
    )

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

    intervals: dict[str, int] | None = serializable_field(
        default=None,
        loading_fn=_intervals_loading_fn,
    )

    intervals_count: dict[str, int] | None = serializable_field(
        default=None,
        loading_fn=lambda data: data.get("intervals_count", None),
    )

    def get_intervals(
        self,
        dataset_n_samples: int | None = None,
        use_defaults_if_missing: bool = True,
        mod_batch_size: bool = True,
    ) -> dict[str, int | float]:
        """get the intervals"""

        # handle the case where both are missing
        if (self.intervals is None) and (self.intervals_count is None):
            if use_defaults_if_missing:
                self.intervals_count = _DEFAULT_INTERVAL_COUNTS()
            else:
                raise ValueError(
                    "both `intervals` and `intervals_count` are missing, and `use_defaults_if_missing` is False. Don't know what intervals to use!"
                )

        # checks
        intervals_new: dict[str, int | float]
        try:
            match (self.intervals is not None, self.intervals_count is not None):
                case (False, False):
                    raise ValueError(
                        "both `intervals` and `intervals_count` are None! this state should be inaccessible"
                    )
                case (True, True):
                    raise ValueError(
                        "both `intervals` and `intervals_count` are specified, this is not allowed!"
                    )
                case (True, False):
                    intervals_new = self.intervals
                case (False, True):
                    if isinstance(dataset_n_samples, int):
                        intervals_new = {
                            k: (
                                int(dataset_n_samples / v)
                                if v > 0
                                else float("inf")
                                # setting a count to < 0 means "dont do it"
                            )
                            for k, v in self.intervals_count.items()
                        }
                    else:
                        raise ValueError(
                            f"{dataset_n_samples = }, but we need an integer to compute the intervals"
                        )

        except ValueError as e:
            _debug_vals: str = (
                f"{dataset_n_samples=}, {use_defaults_if_missing=}, {mod_batch_size=},\n{self.intervals=},\n{self.intervals_count=}"
            )
            raise ValueError(f"{_debug_vals}\ntriggered error:\n{e}") from e

        # disable if set to 0 or negative
        intervals_new = {
            k: (
                v
                if v > 0
                else float("inf")  # mod by infinity is always the number itself
            )
            for k, v in intervals_new.items()
        }

        # check all expected keys are present
        for k in _DEFAULT_INTERVAL_COUNTS().keys():
            if k not in intervals_new:
                raise ValueError(f"missing key {k} in {intervals_new = }")

        # actually return the intervals
        if mod_batch_size:
            return {
                k: (
                    max(1, v // self.batch_size) if isinstance(v, int) else v
                )  # if float, leave it as is since its float("inf")
                for k, v in intervals_new.items()
            }
        else:
            return intervals_new

    def summary(self) -> dict:
        """return a human-readable summary of the config"""
        return dict(
            name=self.name,
            optimizer=self.optimizer.__name__,
            optimizer_kwargs=self.optimizer_kwargs,
            batch_size=self.batch_size,
            dataloader_cfg=self.dataloader_cfg,
            intervals=self.intervals,
            intervals_count=self.intervals_count,
            evals_max_new_tokens=self.evals_max_new_tokens,
            validation_dataset_cfg=(
                self.validation_dataset_cfg
                if (
                    isinstance(self.validation_dataset_cfg, int)
                    or self.validation_dataset_cfg is None
                )
                else self.validation_dataset_cfg.summary()
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
        intervals_count=dict(
            print_loss=100,
            checkpoint=2,
            eval_fast=4,
            eval_slow=2,
        ),
        validation_dataset_cfg=1,
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
        validation_dataset_cfg=10,
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
        validation_dataset_cfg=10,
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
        validation_dataset_cfg=50,
    ),
]


TRAINING_CONFIGS: dict[str, TrainConfig] = {
    cfg.name: cfg for cfg in _TRAINING_CONFIG_LIST
}


def _load_maze_tokenizer(data: dict) -> MazeTokenizer:
    """load the maze tokenizer, including vocab size from a legacy config"""
    if "maze_tokenizer" in data:
        # new style tokenizer
        return load_item_recursive(data["maze_tokenizer"], path=tuple("maze_tokenizer"))
    else:
        if "token_arr" in data["dataset_cfg"]:
            output: MazeTokenizer = MazeTokenizer(
                tokenization_mode=TokenizationMode.AOTP_UT_rasterized,
                max_grid_size=None,
            )
        else:
            raise ValueError("Could not find vocab size in legacy config")


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
    maze_tokenizer: MazeTokenizer | None = serializable_field(
        default_factory=lambda: None,
        loading_fn=_load_maze_tokenizer,
    )

    _tokenizer: PreTrainedTokenizer | None = serializable_field(
        default=None,
        serialization_fn=lambda x: str(x),
        loading_fn=lambda data: None,
    )

    # shortcut properties
    @property
    def d_model(self) -> int:
        return self.model_cfg.d_model

    @property
    def d_head(self) -> int:
        return self.model_cfg.d_head

    @property
    def n_layers(self) -> int:
        return self.model_cfg.n_layers

    @property
    def n_heads(self) -> int:
        return self.model_cfg.n_heads

    def _set_tok_gridsize_from_dataset(self):
        self.maze_tokenizer.max_grid_size = self.dataset_cfg.max_grid_n
        self.maze_tokenizer.clear_cache()

    def __post_init__(self):
        # fallback to default maze tokenizer if no kwargs are provided
        if self.pretrainedtokenizer_kwargs is None:
            if self.maze_tokenizer is None:
                # TODO: is this the right default? maybe set it to AOTP_UT_rasterized
                # since thats what legacy models are likely to be?
                self.maze_tokenizer = MazeTokenizer(
                    tokenization_mode=TokenizationMode.AOTP_UT_uniform,
                    max_grid_size=None,
                )

        # update the config of the maze tokenizer if there is no grid size
        # since we need the token array for the vocab size of the model
        if self.maze_tokenizer is not None:
            if self.maze_tokenizer.max_grid_size is None:
                self._set_tok_gridsize_from_dataset()

    def summary(self) -> str:
        return {
            "name": self.name,
            "dataset_cfg": self.dataset_cfg.summary(),
            "model_cfg": self.model_cfg.summary(),
            "train_cfg": self.train_cfg.summary(),
            "pretrainedtokenizer_kwargs": self.pretrainedtokenizer_kwargs,
            "maze_tokenizer": (
                self.maze_tokenizer.summary()
                if self.maze_tokenizer is not None
                else None
            ),
        }

    @property
    def seed(self) -> int:
        return self.dataset_cfg.seed

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """get a tokenizer via a pretrainedtokenizer_kwargs, or a hugging maze tokenizer"""
        if self._tokenizer is None:
            if self.pretrainedtokenizer_kwargs is not None:
                raise ValueError(
                    "Obsolete tokenizer initialization, caller should revise `ConfigHolder` initialization."
                )
            elif self.maze_tokenizer is not None:
                return HuggingMazeTokenizer(
                    seq_len_max=self.dataset_cfg.seq_len_max,
                    maze_tokenizer=self.maze_tokenizer,
                    name_or_path=(
                        "hugging_maze_tokenizer"
                        if self.maze_tokenizer is None
                        else f"hugging_maze_tokenizer{self.maze_tokenizer.name}"
                    ),
                )
            else:
                raise ValueError("no tokenizer specified")
        return self._tokenizer

    @cached_property
    def hooked_transformer_cfg(self) -> HookedTransformerConfig:
        return HookedTransformerConfig(
            act_fn=self.model_cfg.act_fn,
            d_model=self.model_cfg.d_model,
            d_head=self.model_cfg.d_head,
            n_layers=self.model_cfg.n_layers,
            positional_embedding_type=self.model_cfg.positional_embedding_type,
            n_ctx=self.dataset_cfg.seq_len_max,
            d_vocab=self.maze_tokenizer.vocab_size,
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
    @dynamic_docstring(
        dataset_cfg_names=str(list(MAZE_DATASET_CONFIGS.keys())),
        model_cfg_names=str(list(GPT_CONFIGS.keys())),
        train_cfg_names=str(list(TRAINING_CONFIGS.keys())),
    )
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
        """

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
            try:
                config = ConfigHolder(
                    name=name,
                    dataset_cfg=MAZE_DATASET_CONFIGS[dataset_cfg_name],
                    model_cfg=GPT_CONFIGS[model_cfg_name],
                    train_cfg=TRAINING_CONFIGS[train_cfg_name],
                )
            except KeyError as e:
                raise KeyError(
                    "tried to get a config that doesn't exist, check the names.\n",
                    f"{dataset_cfg_name = }, {model_cfg_name = }, {train_cfg_name = }\n",
                    ConfigHolder.get_config_multisource.__doc__,
                ) from e

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
class ZanjHookedTransformer(ConfiguredModel[ConfigHolder], HookedTransformer):
    """A hooked transformer that is configured by a ConfigHolder

    the inheritance order is critical here -- super() does not call parent, but calls the next class in the MRO
    So, we need ConfiguredModel to take the ConfigHolder and pass kwargs to HookedTransformer
    """

    def __init__(self, cfg_holder: ConfigHolder) -> None:
        super(ZanjHookedTransformer, self).__init__(
            # for ConfiguredModel
            zanj_model_config=cfg_holder,
            # for HookedTransformer
            cfg=cfg_holder.hooked_transformer_cfg,
            tokenizer=cfg_holder.tokenizer,
        )

        # update the tokenizer attributes (evil)
        # see `apply_overrides()` code for info
        if isinstance(self.zanj_model_config.tokenizer, HuggingMazeTokenizer):
            self.zanj_model_config.tokenizer.apply_overrides()
            self.set_tokenizer(
                self.zanj_model_config.tokenizer,
                default_padding_side=self.zanj_model_config.tokenizer.padding_side,
            )
        else:
            warnings.warn(
                "tokenizer is not a HuggingMazeTokenizer, so we can't apply overrides. this might break padding and your whole model"
            )

    @property
    def config(self) -> ConfigHolder:
        return self.zanj_model_config

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
        self.zanj_model_config.model_cfg.weight_processing["are_weights_processed"] = (
            self.zanj_model_config.model_cfg.weight_processing["are_weights_processed"]
            or (not recover_exact)
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
            # move_state_dict_to_device=not recover_exact, # this no longer works as of TransformerLens 1.4.0 but seemed to work previously?
        )
        self.setup()  # Re-attach layernorm hooks by calling setup
        self.eval()
