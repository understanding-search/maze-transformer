from functools import cache
import os
from datetime import datetime
import json
from pathlib import Path
from typing import Annotated, Callable, Any, NamedTuple
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTConfig
from muutils.logger import Logger, TimerContext
from muutils.json_serialize import json_serialize, dataclass_serializer_factory
from muutils.misc import sanitize_fname

from maze_transformer.tokenizer import DatasetConfig

TokenizerFunction = Callable[[list[str]], list[int]]

@dataclass(frozen=True, kw_only=True)
class BaseGPTConfig:
	"""gpt model config without vocab size, context size, or padding token"""
	n_embed: int = 128
	n_layer: int = 4
	n_head: int = 4
	_kwargs: dict = field(default_factory=dict)

	def as_dict(self) -> dict:
		return dict(
			**{
				k: v 
				for k, v in self.__dataclass_fields__.items() 
				if k != "_kwargs"
			},
			**self._kwargs,
		)

@dataclass(kw_only=True)
class TrainConfig:
	"""full training configuration"""
	name: str
	jsonl_path: Annotated[Path, "jsonl file to read for training data"]
	jsonl_tokenized: Annotated[bool, "whether the jsonl data is already tokenized"] = True
	tokenizer: Annotated[TokenizerFunction|None, "if jsonl is not yet tokenized, provide a tokenizing function"] = None

	base_config: BaseGPTConfig = field(default_factory=BaseGPTConfig)
	# wandb_proj: str|None = None
	device: Annotated[torch.device, "device to use for training"] = torch.device(
		"cuda" if torch.cuda.is_available() else "cpu"
		# "cpu"
	)

	max_samples: int|None = None
	epochs: int = 1
	optimizer: torch.optim.Optimizer = torch.optim.RMSprop
	optimizer_kwargs: dict[str, Any] = field(
		default_factory = lambda : dict(lr = 0.0001)
	)
	batch_size: int = 256

	dataloader_cfg: dict = field(default_factory = lambda : dict(
		shuffle = False, # TODO: change this to true if shuffling removed from `chop_sequences.py`
		num_workers = 0, # make this smaller if you're not running on a big cluster probably
		persistent_workers = False,
		drop_last = True,
		# collate_fn = None, # we pad the tensors in the Dataset object
		# batch_size = None, # see batchsize in the encompassing TrainConfig
	))

	n_prints_thru_training: int = 100
	checkpoint_interval_sequences: int = 100000

	# n_ctx: int = property(lambda self: self.model_config.n_ctx)
	
	def get_gpt_config(
			self, 
			**kwargs,
		) -> OpenAIGPTConfig:

		return OpenAIGPTConfig(
			**dict(
				**self.base_config.as_dict(),
				**kwargs,
			)
		)

TrainConfig.serialize = dataclass_serializer_factory(
	TrainConfig,
	special_serializers=dict(
		gpt_config = lambda self: json_serialize(self.gpt_config().to_dict()),
		_optimizer_name = lambda self: self.optimizer.__name__,
		base_config = lambda self: self.base_config.as_dict,
	),
	fields_exclude=["optimizer"],
)


TrainingSetup = NamedTuple("TrainingSetup", [
	("data_cfg", DatasetConfig),
	("train_cfg", TrainConfig),
	("model_cfg", OpenAIGPTConfig),
	("logger", Logger),
	("basepath_train", Path),
])

def setup_train(
		basepath: Path, 
		train_cfg: TrainConfig,
		data_cfg_class: type = DatasetConfig,
		data_cfg_fname: str = "cfg.json",
		**cfg_kwargs,
	) -> tuple[DatasetConfig, Logger, Path]:

	basepath = Path(basepath)

	# load the dataset config
	cfg_path: Path = Path(basepath) / data_cfg_fname
	with open(cfg_path, "r") as f:
		data_cfg: data_cfg_class = data_cfg_class.load(json.load(f))
	data_cfg.name = f"{sanitize_fname(data_cfg.name)}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

	# set up paths
	basepath_train: Path = basepath / data_cfg.name
	os.makedirs(basepath_train, exist_ok = True)
	with open(basepath_train / "config.json", "w") as f:
		json.dump(json_serialize(data_cfg), f, indent = "\t")

	# set up logger
	logger: Logger = Logger(
		log_path=Path(basepath_train / "log.jsonl").as_posix(),
		console_print_threshold=30,
		stream_default_levels={"log_config": 40, "train": 50},
	)

	# set up the training config
	model_cfg: OpenAIGPTConfig = train_cfg.get_gpt_config(
		**data_cfg.gpt_config_kwargs,
	)

	logger.log("loaded data config, initialized logger")
	logger.log_config(dict(data_cfg = json_serialize(data_cfg)))
	logger.log_config(dict(train_cfg = json_serialize(train_cfg)))
	logger.log_config(dict(model_cfg = json_serialize(model_cfg)))
	logger.log_config(dict(logger_cfg =
		{
			"data_cfg.name": data_cfg.name,
			"train_cfg.name": train_cfg.name,
			"basepath": basepath,
			"basepath_train": basepath_train,
			"log_file": logger._log_path,
		},
		lvl = 10,
	))

	return TrainingSetup(
		data_cfg = data_cfg,
		train_cfg = train_cfg,
		model_cfg = model_cfg,
		logger = logger,
		basepath_train = basepath_train,
	)
		


def train(
	basepath: Path, 
	train_cfg: TrainConfig,
	data_cfg_class: type = DatasetConfig,
	data_cfg_fname: str = "cfg.json",
	**cfg_kwargs,
) -> None:
	
	# setup paths and logger
	# ==================================================
	training_setup: TrainingSetup = setup_train(
		basepath=basepath,
		train_cfg=train_cfg,
		data_cfg_class=data_cfg_class,
		data_cfg_fname=data_cfg_fname,
		**cfg_kwargs,
	)
	data_cfg: DatasetConfig = training_setup.data_cfg
	model_cfg: OpenAIGPTConfig = training_setup.model_cfg
	logger: Logger = training_setup.logger
	basepath_train: Path = training_setup.basepath_train

	logger.log("finished setting up paths and logger")

	logger.log("load, process, and batch")
	# ==================================================
	logger.log("loading Dataset", 10)
	dataset: data_cfg_class = data_cfg_class.disk_load(
		path_base = basepath,
		do_tokenized = True,
	)

	logger.log_elapsed_last()
	logger.log(f"loaded {len(dataset)} sequences", 20)
	logger.log("creating dataloader", 10)
	dataloader: DataLoader = DataLoader(
		dataset, 
		batch_size = train_cfg.batch_size,
		**train_cfg.dataloader_cfg,
	)
	logger.log_elapsed_last()


	logger.log("initialize the model and optimizer")
	# ==================================================
	logger.log("initializing model", 10)
	model: OpenAIGPTLMHeadModel = OpenAIGPTLMHeadModel(model_cfg).to(model_cfg.device)
	logger.log_elapsed_last()
	logger.log("initializing optimizer", 10)
	optimizer: torch.optim.Optimizer = model_cfg.optimizer(model.parameters(), **model_cfg.optimizer_kwargs)
	logger.log_elapsed_last()

	# train the model
	# ==================================================
	if train_cfg.epochs > 1:
		raise NotImplementedError("multiple epochs not implemented, get more data instead")

	model.train()
	logger.log("starting training")
	n_batches: int = len(dataloader)
	logger.log({"n_batches": n_batches}, 10)
	print_every_iter: int = n_batches // cfg.n_prints_thru_training

	for iteration, (batch, labels) in enumerate(dataloader):

		# compute loss
		with TimerContext() as timer_loss:
			output = model(
				batch,
				labels=labels,
				# with_backward=True, 
				# keep_outputs=False,
			)
			loss = output.loss
			loss.backward()

		# optimize
		with TimerContext() as timer_optim:
			optimizer.step()
			optimizer.zero_grad()

		# logging
		log_data: dict[str, Any] = json_serialize({
			"iteration": iteration,
			"total_sequences": iteration * train_cfg.batchsize,
			"loss": loss,
			# "train/grad_norm": output.grad_norm,
			"timer_loss": timer_loss.elapsed_time,
			"timer_optim": timer_optim.elapsed_time,
			"time_current": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		})

		logger.train(
			log_data, 
			console_print = (iteration % print_every_iter == 0) or (iteration == 0)
		)

		if iteration % train_cfg.checkpoint_interval_sequences == 0:
			logger.log("saving model", 10)
			torch.save(model.state_dict(), basepath_train / f"model.iter_{iteration}.pt")
			logger.log_elapsed_last()


	# save the model
	# ==================================================
	torch.save(model.state_dict(), basepath_train / "model.pt")


if __name__ == "__main__":
	import fire
	fire.Fire(train)

