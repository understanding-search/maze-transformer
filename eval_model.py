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
from muutils.tensor_utils import ATensor
from muutils.statcounter import StatCounter
from muutils.misc import shorten_numerical_to_str

from maze_transformer.training.config import TrainConfig
from maze_transformer.training.dataset import GPTDatasetConfig
from maze_transformer.training.training import TRAIN_SAVE_FILES


def check_configs_present(folder: Path) -> bool:
	return (
		(folder / TRAIN_SAVE_FILES.data_cfg).exists()
		and (folder / TRAIN_SAVE_FILES.train_cfg).exists()
	)

def load_model_with_configs(model_path: str, data_cfg_class: type) -> tuple[OpenAIGPTLMHeadModel, TrainConfig, GPTDatasetConfig]:
	"""
	Load a model and associated config files from a path.
	"""

	# TODO: make this less fragile
	# load the configs
	# get path to the folder containing the model
	model_folder: Path = Path(model_path).parent
	# check for the filenames, go up a dir if they don't exist
	if not check_configs_present(model_folder):
		model_folder = model_folder.parent
		assert check_configs_present(model_folder), f"Couldn't find configs in directory of or parent directory of {model_path}"

	# load the configs
	train_cfg: TrainConfig = TrainConfig.load(
		json.loads((model_folder / TRAIN_SAVE_FILES.train_cfg).read_text()),
	)
	data_cfg: GPTDatasetConfig = data_cfg_class.load(
		json.loads((model_folder / TRAIN_SAVE_FILES.data_cfg).read_text())
	)


	model = OpenAIGPTLMHeadModel(OpenAIGPTConfig(**model_cfg_inputs))
	state_dict = torch.load(model_path)
	print(state_dict.keys())
	model.load_state_dict(state_dict)
	model.eval()
	print(f"loaded model with {shorten_numerical_to_str(model.num_parameters())} parameters")
	return model


def predict_tokens(model: OpenAIGPTLMHeadModel, inputs: ATensor, n_tokens: int = 32, **generate_kwargs):
	"""
	Predict the next token.
	"""
	with torch.no_grad():
		predictions = model.generate(inputs, max_length=n_tokens, min_length=n_tokens, **generate_kwargs)
	return predictions


def plot_predicted_path(
		model_path: str,





