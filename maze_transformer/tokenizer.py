import json
import os
import sys
import inspect
from functools import cached_property
from itertools import chain, product
from typing import Any, Callable, Generic, Literal, NamedTuple, Sequence, TypeVar, Union
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from muutils.tensor_utils import ATensor, NDArray, DTYPE_MAP, lpad_array
from muutils.json_serialize import json_serialize, dataclass_serializer_factory, dataclass_loader_factory, try_catch, JSONitem
from muutils.misc import freeze

from maze_transformer.latticemaze import LatticeMaze, Coord, CoordTup, CoordArray
from maze_transformer.generators import LatticeMazeGenerators, GENERATORS_MAP

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
			node_token_map[tuple(self.pos_start)],
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


@dataclass(frozen=True, kw_only=True)
class MazeDatasetConfig:
	"""maze dataset configuration, including tokenizers"""
	grid_n: int
	n_mazes: int
	grid_shape = property(lambda self: (self.grid_n, self.grid_n))
	maze_ctor: Callable = LatticeMazeGenerators.gen_dfs
	device: torch.device = field(
		default_factory = lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
	)
	# paths_per_maze: int = 5,
	# p_min_tgt_dist: float = 0.2,
	dtype: torch.dtype|np.dtype = field(default_factory=lambda : np.int16)

	seq_len_min: int = 1
	seq_len_max: int = 2048

	@cached_property
	def node_token_map(self) -> dict[CoordTup, str]:
		"""map from node to token"""
		return {
			tuple(c): f"({c[0]},{c[1]})"
			for c in np.ndindex(self.grid_shape)
		}

	@cached_property
	def token_arr(self) -> list[str]:
		"""map from index to token"""
		return [
			*list(SPECIAL_TOKENS.values()),
			*list(self.node_token_map.values()),
		]

	@cached_property
	def tokenizer_map(self) -> dict[str, int]:
		"""map from token to index"""
		return {
			t: i
			for i, t in enumerate(self.token_arr)
		}

	def tokenize_seq(self, seq: list[str], pad_len: bool|int = True) -> NDArray:
		"""tokenize sequence"""
		if pad_len:
			# adjust pad length
			if isinstance(pad_len, bool):
				pad_len = self.seq_len_max
			elif isinstance(pad_len, int):
				assert pad_len >= len(seq)
			else:
				raise TypeError(f"pad_len must be bool or int, not {type(pad_len) = } {pad_len = }")
			
			return lpad_array(
				np.array(
					[ self.tokenizer_map[t] for t in seq ], 
					dtype=self.dtype,
				),
				padded_length=pad_len,
				pad_value=self.tokenizer_map[SPECIAL_TOKENS["padding"]],
			)

		else:
			return np.array(
				[ self.tokenizer_map[t] for t in seq ], 
				dtype=self.dtype,
			)

	def serialize(self) -> JSONitem:

		maze_ctor: dict = { "__name__": self.maze_ctor.__name__ }
		try:
			maze_ctor["code_hash"] = hash(inspect.getsource(self.maze_ctor))
			maze_ctor["sourcefile"] = inspect.getsourcefile(self.maze_ctor)
		except (TypeError, OSError) as e:
			print(e, file=sys.stderr)
			maze_ctor["code_hash"] = None
			maze_ctor["sourcefile"] = None
			maze_ctor["__exception__"] = str(e)

		return dict(
			grid_n = self.grid_n,
			n_mazes = self.n_mazes,
			grid_shape = self.grid_shape,
			maze_ctor = maze_ctor,
			device = str(self.device),
			dtype = str(self.dtype),
			node_token_map = json_serialize(self.node_token_map),
			token_arr = json_serialize(self.token_arr),
			tokenizer_map = json_serialize(self.tokenizer_map),
		)

	@classmethod
	def load(cls, data: JSONitem) -> "MazeDatasetConfig":
		output = cls(
			grid_n = data["grid_n"],
			n_mazes = data["n_mazes"],
			maze_ctor = GENERATORS_MAP[data["maze_ctor"]["__name__"]],
			device = torch.device(data["device"]),
			dtype = DTYPE_MAP[data["dtype"]],
		)

		# validate
		assert output.grid_shape == data["grid_shape"]
		assert output.node_token_map == data["node_token_map"]
		assert output.token_arr == data["token_arr"]
		assert output.tokenizer_map == data["tokenizer_map"]

		assert output.maze_ctor.__name__ == data["maze_ctor"]["__name__"]
		
		if hash(inspect.getsource(output.maze_ctor)) != data["maze_ctor"]["code_hash"]:
			print(f"WARNING: code hash mismatch for maze_ctor {output.maze_ctor.__name__}", file=sys.stderr)
		
		if inspect.getsourcefile(output.maze_ctor) != data["maze_ctor"]["sourcefile"]:
			print(f"WARNING: sourcefile mismatch for maze_ctor {output.maze_ctor.__name__}", file=sys.stderr)

		return output


class MazeDataset(Dataset):
	"""maze dataset"""

	def __init__(
			self, 
			cfg: MazeDatasetConfig, 
			mazes_objs: list[SolvedMaze]|None = None,
			mazes_tokens: list[list[str]]|None = None,
			mazes_tokenized: list[NDArray[("sequence", "tokens")]]|None = None,
			paths: dict[str, str] = None,
		) -> None:
		super().__init__()

		self.cfg: MazeDatasetConfig = cfg

		# get mode
		if sum(
			1 if x is None else 0  
			for x in [mazes_objs, mazes_tokens, mazes_tokenized]
		) < 1:
			raise ValueError("at least one of mazes_objs, mazes_tokens, mazes_tokenized must be provided to MazeDataset")

		# transfer
		self.mazes_objs: list[SolvedMaze]|None = mazes_objs
		self.mazes_tokens: list[list[str]]|None = mazes_tokens
		self.mazes_tokenized: list[NDArray[("sequence", "tokens")]]|None = mazes_tokenized

		# process into tokens
		# TODO: parallelize this
		if (self.mazes_objs is not None) and (self.mazes_tokens is None):
			self.mazes_tokens = [ m.as_tokens(cfg.node_token_map) for m in self.mazes_objs ]

		# process tokens into tensors
		# TODO: parallelize this
		if (self.mazes_tokens is not None) and (mazes_tokenized is None):
			max_len: int = max(len(t) for t in self.mazes_tokens)
			if max_len > cfg.seq_len_max:
				raise ValueError(f"{max_len=} exceeds {cfg.seq_len_max=}")
			
			self.mazes_tokenized = np.array([
				cfg.tokenize_seq(m, pad_len = cfg.seq_len_max)
				for m in self.mazes_tokens
			])

		# validate
		if any(x is None for x in (self.mazes_objs, self.mazes_tokens, self.mazes_tokenized)):
			raise ValueError(f"MazeDataset invalid, something is None: {type(self.mazes_objs) = } {type(self.mazes_tokens) = } {type(self.mazes_tokenized) = }")
		
		if len(self.mazes_objs) != len(self.mazes_tokens):
			raise ValueError(f"MazeDataset invalid: {len(self.mazes_objs) = }, {len(self.mazes_tokens) = }")
		
		if len(self.mazes_objs) != len(self.mazes_tokenized):
			raise ValueError(f"MazeDataset invalid: {len(self.mazes_objs) = }, {len(self.mazes_tokenized) = }")

	def __getitem__(self, idx: int) -> NDArray[("tokens")]:
		return self.mazes_tokenized[idx]

	@classmethod
	def gen_default(
		cls,
		cfg: MazeDatasetConfig,
	) -> "MazeDataset":
		"""generate a dataset of mazes
		
		p_min_tgt_dist is the minimum manhattan distance between the start and target,
		as a fraction of max of the maze's dimensions
		"""

		# TODO: minimum separation
		# n_min_tgt_dist: int = int(max(maze.grid_shape) * p_min_tgt_dist)

		"""if np.abs(start_node - end_node).sum() < n_min_tgt_dist:
			# if too close, move end node towards the corner opposite the start node
			opposite_corner: CoordTup = (
				maze.grid_shape[0] * round(start_node[0] / maze.grid_shape[0]),
				maze.grid_shape[1] * round(start_node[1] / maze.grid_shape[1]),
			)
			# end_node +=
		"""

		mazes: list[SolvedMaze] = list()
		endpoint_nodes: NDArray[(("maze_idx", cfg.n_mazes), ("start_end", 2), ("coord", 2)), np.int8] = np.random.randint(0, cfg.grid_shape, (cfg.n_mazes, 2, 2))

		print(endpoint_nodes)

		for i, (c_start, c_end) in enumerate(endpoint_nodes):
			m: LatticeMaze = cfg.maze_ctor(cfg.grid_shape)
			path: CoordArray = np.array(m.find_shortest_path(c_start, c_end))
			mazes.append(SolvedMaze(
				maze=m,
				solution=path,
			))

		return cls(
			cfg=cfg,
			mazes_objs=mazes,
		)
		

	def serialize_config(self) -> JSONitem:
		return json_serialize(self.cfg)

	@freeze
	class DISK_SAVE_FILES:
		cfg: str = "cfg.json"
		obj: str = "maze_obj.jsonl"
		tokens: str = "maze_tokens.jsonl"
		tokenized: str = "maze_tokenized.npy"

	def disk_save(
			self, 
			path_base: str = "data/test-001",
			do_config: bool = True,
			do_obj: bool = False,
			do_tokens: bool = False,
			do_tokenized: bool = True,
		) -> None:

		# make the appropriate directories
		print(f"saving to '{path_base}'")
		os.makedirs(path_base, exist_ok = True)

		if do_config:
			# save config as json
			with open(f"{path_base}/{self.DISK_SAVE_FILES.cfg}", "w") as f:
				json.dump(json_serialize(self.cfg), f)

		if do_obj:
			raise NotImplementedError("do_obj not implemented")

		if do_tokens:
			raise NotImplementedError("do_tokens not implemented")


		if do_tokenized:
			# save tokenized data as npz
			np.save(f"{path_base}/{self.DISK_SAVE_FILES.tokenized}", self.mazes_tokenized)

	@classmethod
	def disk_load(
		cls,
		path_base: str,
					do_config: bool = True,
			do_obj: bool = False,
			do_tokens: bool = False,
			do_tokenized: bool = True,
		) -> "MazeDataset":

		if do_obj:
			raise NotImplementedError("do_obj not implemented")
		if do_tokens:
			raise NotImplementedError("do_tokens not implemented")

		if do_config:
			# load config from json
			with open(f"{path_base}/{cls.DISK_SAVE_FILES.cfg}", "r") as f:
				cfg: MazeDatasetConfig = MazeDatasetConfig.load(json.load(f))
			
		if do_tokenized:
			# load tokenized data from npz
			tokenized: NDArray = np.load(f"{path_base}/{cls.DISK_SAVE_FILES.tokenized}")

		return cls(
			cfg = cfg if do_config else None,
			mazes_objs = None, # if do_obj else None,
			mazes_tokens = None, # if do_tokens else None,
			mazes_tokenized = tokenized if do_tokenized else None,
		)



			









