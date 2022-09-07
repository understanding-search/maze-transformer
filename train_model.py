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

from maze_transformer.training.tokenizer import DatasetConfig, MazeDataset, MazeDatasetConfig
from maze_transformer.training.training import TokenizerFunction, TrainConfig, setup_train, TrainingSetup



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
	tracemalloc.start()
	dataset: data_cfg_class = MazeDatasetConfig._dataset_class.disk_load(
		path_base = basepath,
		do_config = True,
		do_tokenized = True,
	)
	logger.log_elapsed_last()
	logger.mem_usage()
	length_stats: StatCounter = StatCounter(dataset.get_all_lengths().tolist())
	logger.log({"dataset_seq_len_stats": length_stats.summary()})
	logger.log({"dataset_seq_len_stats": length_stats.serialize()}, lvl=50)

	logger.log(f"loaded {len(dataset)} sequences", 20)
	logger.log("creating dataloader", 10)
	dataloader: DataLoader = DataLoader(
		dataset, 
		batch_size = train_cfg.batch_size,
		**train_cfg.dataloader_cfg,
	)
	logger.log_elapsed_last()
	logger.mem_usage()


	logger.log("initialize the model and optimizer")
	# ==================================================
	logger.log("initializing model", 10)
	model: OpenAIGPTLMHeadModel = OpenAIGPTLMHeadModel(model_cfg).to(model_cfg.device)
	logger.log_elapsed_last()
	logger.mem_usage()
	logger.log({"model_cfg.device": model_cfg.device, "model.device": model.device}, 20)

	logger.log("initializing optimizer", 10)
	optimizer: torch.optim.Optimizer = train_cfg.optimizer(
		model.parameters(), 
		**train_cfg.optimizer_kwargs,
	)
	logger.log_elapsed_last()
	logger.mem_usage()
	model_n_params:int = sum(p.numel() for p in model.parameters() if p.requires_grad)
	logger.log(dict(model_n_params = model_n_params), 20)

	# train the model
	# ==================================================
	if train_cfg.epochs > 1:
		raise NotImplementedError("multiple epochs not implemented, get more data instead")

	model.train()
	logger.log("starting training")
	n_batches: int = len(dataloader)
	logger.log({"n_batches": n_batches}, 10)

	n_sequences: int
	print_loss_interval_iters: int = int(train_cfg.print_loss_interval // train_cfg.batch_size)
	checkpoint_interval_iters: int = int(train_cfg.checkpoint_interval // train_cfg.batch_size)
	for iteration, batch in enumerate(dataloader):

		# compute loss
		with TimerContext() as timer_loss:
			batch_on_device: ATensor[("batch", "sequence")] = batch.type(dtype=torch.LongTensor).to(model.device)
			# logger.tensor_dims({
			# 	"batch_on_device.shape" : batch_on_device.shape, 
			# 	"batch_on_device.dtype" : str(batch_on_device.dtype),
			# 	"batch_on_device.device" : str(batch_on_device.device),
			# }, lvl = 20)

			output = model(
				batch_on_device[:, :-1],
				labels=batch_on_device[:, 1:],
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
		n_sequences = iteration * train_cfg.batch_size
		log_data: dict[str, Any] = json_serialize({
			"iter": iteration,
			"loss": loss,
			# "train/grad_norm": output.grad_norm,
			"n_sequences": n_sequences,
			"time_current": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
			"timer_loss": round(timer_loss.elapsed_time, 6),
			"timer_optim": round(timer_optim.elapsed_time, 6),
		})

		del output
		del loss

		logger.train(
			log_data, 
			lvl=50,
			console_print = (
				(iteration % print_loss_interval_iters == 0) 
				or (iteration % checkpoint_interval_iters == 0) 
			),
		)

		if iteration % checkpoint_interval_iters == 0:
			model_save_path: Path = basepath_train / f"model.iter_{iteration}.pt"
			logger.saving(f"saving model to {model_save_path.as_posix()}", 10)
			torch.save(model.state_dict(), model_save_path)
			logger.log_elapsed_last(stream="saving")


	# save the model
	# ==================================================
	torch.save(model.state_dict(), basepath_train / "model.pt")


def main(basepath: str):
	train_cfg = TrainConfig(
		name = "test",
		base_gpt_cfg = BaseGPTConfig(),
	)


	train(
		basepath = Path(basepath),
		train_cfg = train_cfg,
		dataset_class = MazeDataset,
		data_cfg_class=MazeDatasetConfig,
		data_cfg_fname="cfg.json",
	)


if __name__ == "__main__":
	import fire
	fire.Fire(main)

