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

from maze_transformer.latticemaze import LatticeMaze, Coord, CoordTup, CoordArray
from maze_transformer.generators import LatticeMazeGenerators, GENERATORS_MAP

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


@dataclass(kw_only=True)
class DatasetConfig:
	"""base config class"""
	name: str = "DatasetConfig_default"
	device: torch.device = field(
		default_factory = lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
	)
	dtype: torch.dtype|np.dtype = field(default_factory=lambda : torch.int16)
	seq_len_min: int = 1
	seq_len_max: int = 512

	
	@cached_property
	def token_arr(self) -> list[str]:
		raise NotImplementedError()

	@cached_property
	def padding_token_idx(self) -> str:
		raise NotImplementedError()

	@cached_property
	def tokenizer_map(self) -> dict[str, int]:
		"""map from token to index"""
		return {
			t: i
			for i, t in enumerate(self.token_arr)
		}

	@cached_property
	def gpt_config_kwargs(self) -> dict:
		"""gpt model config with vocab size, context size, and padding token"""
		return dict(
			vocab_size = len(self.token_arr),
			n_positions = self.seq_len_max,
			pad_token_id = self.padding_token_idx, # The id of the _padding_ token.
			bos_token_id = self.padding_token_idx, # The id of the _beginning-of-stream_ token.
			eos_token_id = self.padding_token_idx, # The id of the _end-of-stream_ token.
		)

	def tokenize_seq(self, seq: list[str]) -> ATensor:
		"""tokenize sequence"""
		return torch.tensor(
			[ self.tokenizer_map[t] for t in seq ], 
			dtype=self.dtype,
			device="cpu",
		)

	def serialize(self) -> dict:
		raise NotImplementedError()

	@classmethod
	def load(cls, data: dict) -> "DatasetConfig":
		raise NotImplementedError()


@dataclass(kw_only=True)
class MazeDatasetConfig(DatasetConfig):
	"""maze dataset configuration, including tokenizers"""
	grid_n: int
	n_mazes: int
	grid_shape = property(lambda self: (self.grid_n, self.grid_n))
	maze_ctor: Callable = LatticeMazeGenerators.gen_dfs
	# paths_per_maze: int = 5,
	# p_min_tgt_dist: float = 0.2,

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

	@property
	def n_tokens(self) -> int:
		return len(self.token_arr)

	@cached_property
	def padding_token_idx(self) -> str:
		return self.tokenizer_map[SPECIAL_TOKENS["padding"]]


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
			n_tokens = self.n_tokens,
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
		assert tuple(output.grid_shape) == tuple(data["grid_shape"]), f"{output.grid_shape = } {data['grid_shape'] = }"
		assert json_serialize(output.node_token_map) == data["node_token_map"], f"\n{output.node_token_map = }\n\n{data['node_token_map'] = }"
		assert output.token_arr == data["token_arr"]
		assert output.tokenizer_map == data["tokenizer_map"]

		assert output.maze_ctor.__name__ == data["maze_ctor"]["__name__"]

		assert output.n_tokens == data["n_tokens"]
		
		if hash(inspect.getsource(output.maze_ctor)) != data["maze_ctor"]["code_hash"]:
			print(f"WARNING: code hash mismatch for maze_ctor {output.maze_ctor.__name__}", file=sys.stderr)
		
		if inspect.getsourcefile(output.maze_ctor) != data["maze_ctor"]["sourcefile"]:
			print(f"WARNING: sourcefile mismatch for maze_ctor {output.maze_ctor.__name__}", file=sys.stderr)

		return output



@dataclass(kw_only=True)
class IndexedArray:
	"""join a list of arrays into a single big one with indices
	
	mainly for allowing __getitem__ to work nice for datasets"""
	arr: ATensor
	idxs: ATensor

	def get_len(self, idx: int) -> int:
		return self.idxs[idx+1] - self.idxs[idx]

	def get_all_lengths(self) -> ATensor:
		return torch.cat([
			self.idxs[1:] - self.idxs[:-1],
			torch.tensor([self.arr.shape[0] - self.idxs[-1]], dtype=self.idxs.dtype, device=self.idxs.device),
		])

	@classmethod
	def from_sequences(cls, data: list[ATensor[("tokens")]]) -> "IndexedArray":
		"""process many sequences into a single array, keeping track of sequence start indices
		
		example:
		f( [[a,b,c], [d,e]] ) -> IndexedArray(
			arr = [a,b,c,d,e],
			idxs = [0,3],
		)
		"""
		arr: ATensor = torch.cat(data)
		idxs: ATensor = torch.cumsum( torch.tensor([ 0, *map(len, data) ]), dim = 0 )[:-1]
		return cls(arr=arr, idxs=idxs)

def maze_to_tokens(maze: SolvedMaze, node_token_map: dict[CoordTup, str]) -> list[str]:
	"""convert a maze into a list of tokens"""
	return maze.as_tokens(node_token_map)


class MazeDataset(Dataset):
	"""maze dataset"""

	def __init__(
			self,
			cfg: MazeDatasetConfig, 
			mazes_objs: list[SolvedMaze]|None = None,
			mazes_tokens: list[list[str]]|None = None,
			# mazes_tokenized: list[NDArray[("sequence", "tokens")]]|None = None,
			mazes_array: IndexedArray|None = None,
			paths: dict[str, str] = None,
		) -> None:
		super().__init__()

		self.cfg: MazeDatasetConfig = cfg

		# get mode
		if sum(
			1 if x is None else 0  
			for x in [mazes_objs, mazes_tokens, mazes_array]
		) < 1:
			raise ValueError("at least one of mazes_objs, mazes_tokens, mazes_tokenized must be provided to MazeDataset")

		# transfer
		self.mazes_objs: list[SolvedMaze]|None = mazes_objs
		self.mazes_tokens: list[list[str]]|None = mazes_tokens
		# self.mazes_tokenized: list[NDArray[("sequence", "tokens")]]|None = mazes_tokenized
		self.mazes_array: IndexedArray|None = mazes_array

		# process into tokens
		if (self.mazes_objs is not None) and (self.mazes_tokens is None):
			with multiprocessing.Pool() as pool:
				self.mazes_tokens = list(tqdm(
					pool.imap(
						partial(maze_to_tokens, node_token_map=cfg.node_token_map),
						self.mazes_objs,
					),
					total = len(self.mazes_objs),
					desc = "tokenizing mazes",
					unit = "maze",
				))

		# process tokens into tokenized
		if (self.mazes_tokens is not None) and (mazes_array is None):
			max_len: int = max(len(t) for t in self.mazes_tokens)
			if max_len > cfg.seq_len_max:
				raise ValueError(f"{max_len=} exceeds {cfg.seq_len_max=}")
			
			self.mazes_array = IndexedArray.from_sequences([
				cfg.tokenize_seq(m)
				for m in self.mazes_tokens
			])

		# validate
		# if any(x is None for x in (self.mazes_objs, self.mazes_tokens, self.mazes_tokenized)):
		# 	raise ValueError(f"MazeDataset invalid, something is None: {type(self.mazes_objs) = } {type(self.mazes_tokens) = } {type(self.mazes_tokenized) = }")
		
		if self.mazes_objs is not None and self.mazes_tokens is not None:
			if len(self.mazes_objs) != len(self.mazes_tokens):
				raise ValueError(f"MazeDataset invalid: {len(self.mazes_objs) = }, {len(self.mazes_tokens) = }")
		
		if self.mazes_objs is not None and self.mazes_array.idxs is not None:
			if len(self.mazes_objs) != len(self.mazes_array.idxs):
				raise ValueError(f"MazeDataset invalid: {len(self.mazes_objs) = }, {len(self.mazes_array.idxs) = }")

		if self.mazes_tokens is not None and self.mazes_array.idxs is not None:
			if len(self.mazes_tokens) != len(self.mazes_array.idxs):
				raise ValueError(f"MazeDataset invalid: {len(self.mazes_tokens) = }, {len(self.mazes_array.idxs) = }")

	def __getitem__(self, idx: int) -> ATensor[("tokens")]:
		"""index into mazes_array.arr, getting up to the next sequence start, padding if necessary"""
		# TODO: handling of minimum sequence length

		# last element in mazes_array.idxs whose value is smaller than `idx`
		sequence_idx: int = torch.searchsorted(self.mazes_array.idxs, idx) - 1
		# slice the array from the start of the sequence to `idx`, including `idx`
		end_arr_idx: int = min(
			idx + 1, # up to end of sequence
			self.mazes_array.idxs[sequence_idx] + self.cfg.seq_len_max, # up to sequence length cutoff
		)
		subseq: ATensor = self.mazes_array.arr[ self.mazes_array.idxs[sequence_idx] : end_arr_idx ]
		# left-pad the sequence
		return torch.nn.functional.pad(subseq, (self.cfg.seq_len_max + 1 - len(subseq), 0), value=self.cfg.padding_token_idx)

	def __len__(self) -> int:
		return len(self.mazes_array.arr)

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
		"""namespace for filenames"""
		cfg: str = "cfg.json"
		obj: str = "maze_obj.jsonl"
		tokens: str = "maze_tokens.jsonl"
		tokenized: str = "maze_tokenized.npz"

	def disk_save(
			self, 
			path_base: str = "data/test-001",
			do_config: bool = True,
			do_obj: bool = False,
			do_tokens: bool = True,
			do_tokenized: bool = True,
		) -> None:

		# make the appropriate directories
		print(f"saving to '{path_base}'")
		os.makedirs(path_base, exist_ok = False)

		if do_config:
			# save config as json, with metadata
			config_out: dict[str, JSONitem] = {
				**json_serialize(self.cfg),
				"_postgen_meta": {
					"seq_len_stats": StatCounter(self.mazes_array.get_all_lengths().tolist()).summary(),
				}
			}
			with open(f"{path_base}/{self.DISK_SAVE_FILES.cfg}", "w") as f:
				json.dump(config_out, f, indent="\t")

		if do_obj:
			raise NotImplementedError("do_obj not implemented")

		if do_tokens:
			# save tokens as jsonl
			with open(f"{path_base}/{self.DISK_SAVE_FILES.tokens}", "w") as f:
				for x in self.mazes_tokens:
					f.write(' '.join(x) + "\n")

		if do_tokenized:
			# save tokenized data as npz
			np.savez(
				f"{path_base}/{self.DISK_SAVE_FILES.tokenized}",
				**dict(
					arr=self.mazes_array.arr.cpu().numpy(),
					idxs=self.mazes_array.idxs.cpu().numpy(),
				),
			)

	@classmethod
	def disk_load(
			cls,
			path_base: str,
			do_config: bool = False,
			do_obj: bool = False,
			do_tokens: bool = False,
			do_tokenized: bool = False,
		) -> "MazeDataset":

		cfg: MazeDatasetConfig|None = None
		if do_config:
			# load config from json
			with open(f"{path_base}/{cls.DISK_SAVE_FILES.cfg}", "r") as f:
				cfg = MazeDatasetConfig.load(json.load(f))
		
		mazes_objs: list[SolvedMaze]|None = None
		if do_obj:
			raise NotImplementedError("do_obj not implemented")

		mazes_tokens: list[list[str]]|None = None
		if do_tokens:
			# load tokens from jsonl
			with open(f"{path_base}/{cls.DISK_SAVE_FILES.tokens}", "r") as f:
				mazes_tokens = [
					x.split() 
					for x in f.readlines()
				]
		
		loaded_dict: dict|None = None
		if do_tokenized:
			# load tokenized data from npz
			loaded_dict = np.load(
				f"{path_base}/{cls.DISK_SAVE_FILES.tokenized}",
				allow_pickle=False,
			)

			assert "arr" in loaded_dict
			assert "idxs" in loaded_dict			

		return cls(
			cfg = cfg,
			mazes_objs = mazes_objs,
			mazes_tokens = mazes_tokens,
			mazes_array = None if loaded_dict is None else IndexedArray(
				arr=torch.tensor(loaded_dict["arr"], device="cpu"),
				idxs=torch.tensor(loaded_dict["idxs"], device="cpu"),
			),
		)



MazeDatasetConfig._dataset_class = MazeDataset
			









