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
from maze_transformer.training.config import TRAINING_CONFIGS

from maze_transformer.training.dataset import GPTDatasetConfig, TrainConfig
from maze_transformer.training.mazedataset import MazeDataset, MazeDatasetConfig
from maze_transformer.training.training import TokenizerFunction, TrainConfig, setup_train, TrainingSetup, train, BaseGPTConfig, GPT_CONFIGS



def main(basepath: str, cfg_name: str = "tiny_v1"):

	train_cfg: TrainConfig = TRAINING_CONFIGS[cfg_name]

	train(
		basepath = Path(basepath),
		train_cfg = train_cfg,
		dataset_class = MazeDataset,
		data_cfg_class = MazeDatasetConfig,
	)

if __name__ == "__main__":
	import fire
	fire.Fire(main)

