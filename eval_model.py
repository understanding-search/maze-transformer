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



def load_model(model_path: str):
	"""
	Load a model from a path.
	"""
	model = OpenAIGPTLMHeadModel.from_pretrained(model_path)
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





