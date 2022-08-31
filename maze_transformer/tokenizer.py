import sys
import inspect
from functools import cached_property
from itertools import chain, product
from typing import Any, Callable, Generic, Literal, NamedTuple, Sequence, TypeVar, Union
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from muutils.tensor_utils import ATensor, NDArray, DTYPE_MAP
from muutils.json_serialize import json_serialize, dataclass_serializer_factory, dataclass_loader_factory, try_catch, JSONitem

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
				[ node_token_map[c_s], SPECIAL_TOKENS["connector"], node_token_map[c_e], SPECIAL_TOKENS["adjacency_endline"] ]
				for c_s, c_e in self.maze.as_adjlist()
			]),
			SPECIAL_TOKENS["adjlist_end"],
			# give target
			SPECIAL_TOKENS["target_start"],
			node_token_map[tuple(self.pos_start)],
			SPECIAL_TOKENS["target_end"],
			# give path
			SPECIAL_TOKENS["start_path"],
			*[ node_token_map[c] for c in self.solution ],
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
	dtype: torch.dtype = field(default_factory=torch.int16)

	seq_len_min: int = 1
	seq_len_max: int = 1024

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

	def tokenize_seq(self, seq: list[str]) -> torch.Tensor:
		"""tokenize sequence"""
		return torch.tensor(
			[ self.tokenizer_map[t] for t in seq ], 
			dtype=self.dtype, device=self.device,
		)

	def serialize(self) -> JSONitem:
		return dict(
			grid_n = self.grid_n,
			n_mazes = self.n_mazes,
			grid_shape = self.grid_shape,
			maze_ctor = {
				"__name__": self.maze_ctor.__name__,
				"code_hash": try_catch(lambda x: hash(inspect.getsource(x))),
				"sourcefile": try_catch(lambda x: inspect.getsourcefile(x)),
			},
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

		self.paths: dict[str, str] = paths if paths is not None else dict()

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
			self.mazes_tokenized = [
				cfg.tokenize_seq(m)
				for m in self.mazes_tokens
			]

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
		


	# def 

	# def save_obj(self):
	# 	"""serialize this object"""

	# def save_tokens(self):
	# 	"""serialize this object to tokens"""
	# 	raise NotImplementedError()

	# def save_tokenized(self):


			









