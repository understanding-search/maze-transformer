import json
import os
from pathlib import Path
import sys
import inspect
from functools import cached_property, partial
from itertools import chain, product
from typing import Any, Callable, Generic, Literal, NamedTuple, Sequence, TypeVar, Union
from dataclasses import dataclass, field
import multiprocessing

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import OpenAIGPTConfig
from tqdm import tqdm
from muutils.tensor_utils import ATensor, NDArray, DTYPE_MAP, lpad_array
from muutils.json_serialize import json_serialize, dataclass_serializer_factory, dataclass_loader_factory, try_catch, JSONitem
from muutils.misc import freeze
from muutils.statcounter import StatCounter

from maze_transformer.generation.latticemaze import LatticeMaze, Coord, CoordTup, CoordArray
from maze_transformer.generation.generators import LatticeMazeGenerators, GENERATORS_MAP

# pylint: disable=unused-import

SPECIAL_TOKENS: dict[str, str] = dict(
	adjlist_start = "<ADJLIST_START>",
	adjlist_end = "<ADJLIST_END>",
	target_start = "<TARGET_START>",
	target_end = "<TARGET_END>",
	start_path = "<START_PATH>",
	end_path = "<END_PATH>",
	connector = "<-->",
	adjacency_endline = ";",
	padding = "<PADDING>",
)


@dataclass(frozen=True, kw_only=True)
class SolvedMaze:
	"""solved maze for serialization"""
	maze: LatticeMaze
	solution: CoordArray
	metadata: dict = field(default_factory=dict)
	
	pos_start = property(lambda self: self.solution[0])
	pos_end = property(lambda self: self.solution[-1])

	def as_tokens(self, node_token_map: dict[CoordTup, str]) -> list[str]:
		"""serialize maze and solution to tokens"""
		tokens: list[str] = [
			# give adjacency list
			SPECIAL_TOKENS["adjlist_start"],
			*chain.from_iterable([
				[
					node_token_map[tuple(c_s.tolist())], 
					SPECIAL_TOKENS["connector"], 
					node_token_map[tuple(c_e.tolist())], 
					SPECIAL_TOKENS["adjacency_endline"],
				]
				for c_s, c_e in self.maze.as_adjlist()
			]),
			SPECIAL_TOKENS["adjlist_end"],
			# give target
			SPECIAL_TOKENS["target_start"],
			node_token_map[tuple(self.pos_end)],
			SPECIAL_TOKENS["target_end"],
			# give path
			SPECIAL_TOKENS["start_path"],
			*[ 
				node_token_map[tuple(c.tolist())] 
				for c in self.solution
			],
			SPECIAL_TOKENS["end_path"],
		]

		return tokens






