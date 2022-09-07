from functools import cache
import os
from datetime import datetime
import json
from pathlib import Path
from typing import Annotated, Callable, Any, NamedTuple
from dataclasses import dataclass, field
import tracemalloc


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTConfig
from muutils.logger import Logger, TimerContext
from muutils.json_serialize import json_serialize, dataclass_serializer_factory
from muutils.misc import sanitize_fname
from muutils.tensor_utils import ATensor
from muutils.statcounter import StatCounter


TokenizerFunction = Callable[[list[str]], list[int]]


# ==================================================

@dataclass(frozen=True, kw_only=True)
class BaseGPTConfig:
	"""gpt model config without vocab size, context size, or padding token"""
	gpt_cfg_name: str
	n_embed: int = 128
	n_layer: int = 8
	n_head: int = 4
	_kwargs: dict = field(default_factory=dict)

	def as_dict(self) -> dict:
		return dict(
			**{
				k: getattr(self, k)
				for k in self.__dataclass_fields__ 
				if k != "_kwargs"
			},
			**self._kwargs,
		)

	def serialize(self) -> str:
		return self.as_dict()


# ==================================================

@dataclass(kw_only=True)
class TrainConfig:
	"""full training configuration"""
	name: str

	base_gpt_cfg: BaseGPTConfig
	# wandb_proj: str|None = None
	device: Annotated[torch.device, "device to use for training"] = torch.device(
		# "cuda" if torch.cuda.is_available() else "cpu"
		"cuda:0"
	)

	max_samples: int|None = None
	epochs: int = 1
	optimizer: torch.optim.Optimizer = torch.optim.RMSprop
	optimizer_kwargs: dict[str, Any] = field(
		default_factory = lambda : dict(lr = 0.000001)
	)
	batch_size: int = 128

	dataloader_cfg: dict = field(default_factory = lambda : dict(
		shuffle = True,
		num_workers = 16, # make this smaller if you're not running on a big cluster probably
		persistent_workers = True,
		drop_last = True,
		# collate_fn = None, # we pad the tensors in the Dataset object
		# batch_size = None, # see batchsize in the encompassing TrainConfig
	))

	print_loss_interval: int = 1000
	checkpoint_interval: int = 50000

	seq_len_max: int|None = None

	# n_ctx: int = property(lambda self: self.model_config.n_ctx)
	_gpt_config_ctor_kwargs: dict = field(default_factory=dict)
	
	def get_gpt_config(
			self, 
			**kwargs,
		) -> OpenAIGPTConfig:
		"""passes base_gpt_cfg, device, and _gpt_config_ctor_kwargs to OpenAIGPTConfig"""

		self._gpt_config_ctor_kwargs = {
			**self.base_gpt_cfg.as_dict(),
			**kwargs,
			**dict(device = self.device),
		}

		return OpenAIGPTConfig(**self._gpt_config_ctor_kwargs)

TrainConfig.serialize = dataclass_serializer_factory(
	TrainConfig,
	special_serializers=dict(
		# gpt_config = lambda self: json_serialize(self.get_gpt_config().to_dict()),
		_optimizer_name = lambda self: self.optimizer.__name__,
		base_gpt_cfg = lambda self: self.base_gpt_cfg.as_dict,
	),
	fields_exclude=["optimizer"],
)


# actual configuration setups
# ==================================================

_GPT_CONFIGS_LIST: list[BaseGPTConfig] = [
	BaseGPTConfig(
		gpt_cfg_name = "tiny-v1",
		n_embed=32,
		n_layer=4,
		n_head=2,
	),
	BaseGPTConfig(
		gpt_cfg_name = "medium-v1",
		n_embed=128,
		n_layer=8,
		n_head=4,
	),
]

GPT_CONFIGS: dict[str, BaseGPTConfig] = {
	cfg.gpt_cfg_name: cfg 
	for cfg in _GPT_CONFIGS_LIST
}


_TRAINING_CONFIG_LIST: list[TrainConfig] = [
	TrainConfig(
		name = "tiny-v1",
		base_gpt_cfg = GPT_CONFIGS["tiny-v1"],
		optimizer = torch.optim.RMSprop,
		optimizer_kwargs = dict(lr = 0.000001),
		batch_size = 32,
		dataloader_cfg = dict(
			shuffle = True,
			num_workers = 16, # make this smaller if you're not running on a big cluster probably
			persistent_workers = True,
			drop_last = True,
		),
		print_loss_interval = 1000,
		checkpoint_interval = 50000,
		seq_len_max = 90,
	)
]


TRAINING_CONFIGS: dict[str, TrainConfig] = {
	cfg.name: cfg 
	for cfg in _TRAINING_CONFIG_LIST
}

