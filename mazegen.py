from collections import deque
from functools import cached_property
import inspect
from itertools import chain, product
import random
import sys
import time
from typing import Any, Callable, Generic, Literal, NamedTuple, Sequence, TypeVar, Union
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from muutils.tensor_utils import ATensor, NDArray, DTYPE_MAP
from muutils.json_serialize import json_serialize, dataclass_serializer_factory, dataclass_loader_factory, try_catch, JSONitem
# from muutils.defaulterdict import DefaulterDict


# @dataclass(frozen=True, kw_only=True)
# class Maze:
# 	"""generalized maze class"""
# 	n_nodes: int
	
DIRECTIONS_MAP: NDArray[(("direction", 4), ("axes", 2)), int] = np.array([
	[0, 1], # down
	[0, -1], # up
	[1, 1], # right
	[1, -1], # left
])


NEIGHBORS_MASK: NDArray[(("direction", 4), ("axes", 2)), int] = np.array([
	[0, 1], # down
	[0, -1], # up
	[1, 0], # right
	[-1, 0], # left
])

# print(NEIGHBORS_MASK, NEIGHBORS_MASK.dtype, NEIGHBORS_MASK.shape)

Coord = NDArray[("coord", 2), np.int8]
CoordTup = tuple[int, int]
CoordArray =  NDArray[(("points", Any), ("coord", 2)), np.int8]

# def get_neighbors_2d(c: Coord, maze_shape: tuple[int, int]) -> list[tuple[int, int]]:
# 	"""get the neighbors of a given coordinate"""
# 	neighbors: list[tuple[int, int]] = np.array(c) + NEIGHBORS_MASK

# 	return [
# 		[x, y]
# 		for x, y in neighbors
# 		if (0 <= x < maze_shape[0]) and (0 <= y < maze_shape[1])
# 	]


@dataclass(frozen=True, kw_only=True)
class LatticeMaze:
	"""lattice maze (nodes on a lattice, connections only to neighboring nodes)"""
	lattice_dim: int = 2
	connection_list: NDArray[("lattice_dim", "x", "y"), bool]
	generation_meta: dict|None = None

	grid_shape = property(
		lambda self: self.connection_list.shape[1:]
	)

	n_connections = property(
		lambda self: self.connection_list.sum()
	)


	def as_img(self) -> NDArray[("x", "y"), bool]:

		# set up the background
		print(self.grid_shape)
		img: NDArray[("x", "y"), bool] = np.zeros(
			(
				self.grid_shape[0] * 2 + 1,
				self.grid_shape[1] * 2 + 1,
			),
			dtype=bool,
		)

		# fill in nodes
		img[1::2, 1::2] = True

		# fill in connections
		# print(f"{img[2:-2:2, 1::2].shape = } {self.connection_list[0, :, :-1].shape = }")
		img[2:-2:2, 1::2] = self.connection_list[0, :-1, :]
		img[1::2, 2:-2:2] = self.connection_list[1, :, :-1]

		return img

	def as_adjlist(self, shuffle_d0: bool = True, shuffle_d1: bool = True) -> NDArray[( ("conn", Any), ("start_end", 2), ("coord", 2) ), np.int8]:

		adjlist: NDArray[( ("conn", Any), ("start_end", 2), ("coord", 2) ), np.int8] = np.full(
			(self.n_connections, 2, 2),
			-1,
		)

		if shuffle_d1:
			flip_d1: NDArray[("conn", 1), np.float16] = np.random.rand(self.n_connections)

		# loop over all nonzero elements of the connection list
		for i, (d, x, y) in enumerate(np.ndindex(self.connection_list.shape)):
			if self.connection_list[d, x, y]:
				c_start: CoordTup = (x, y)
				c_end: CoordTup = (
					x + (1 if d == 0 else 0),
					y + (1 if d == 1 else 0),
				)
				adjlist[i, 0] = np.array(c_start)
				adjlist[i, 1] = np.array(c_end)

				# flip if shuffling
				if shuffle_d1 and (flip_d1[i] > 0.5):
						adjlist[i, 0], adjlist[i, 1] = adjlist[i, 1], adjlist[i, 0]					

		if shuffle_d0:
			np.random.shuffle(adjlist)

		return adjlist



	def points_transform_to_img(self, points: CoordArray) -> CoordArray:
		"""transform points to img coordinates"""
		return 2 * points + 1

	@staticmethod
	def heuristic(a: CoordTup, b: CoordTup) -> float:
		"""return manhattan distance between two points"""
		return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])

	def nodes_connected(self, a: Coord, b: Coord, /) -> bool:
		"""returns whether two nodes are connected"""
		delta: Coord = b - a
		if np.abs(delta).sum() > 1:
			# return false if not even adjacent
			return False
		else:
			# test for wall
			dim: int = np.argmax(np.abs(delta))
			clist_node: Coord = a if (delta.sum() > 0) else b
			return self.connection_list[dim, clist_node[0], clist_node[1]]

	def get_coord_neighbors(self, c: Coord) -> CoordArray:
		neighbors: list[Coord] = [
			neighbor
			for neighbor in (c + NEIGHBORS_MASK)
			if (
				(0 <= neighbor[0] < self.grid_shape[0]) # in x bounds
				and (0 <= neighbor[1] < self.grid_shape[1]) # in y bounds
				and self.nodes_connected(c, neighbor) # connected
			)
		]

		return np.array(neighbors)

	def find_shortest_path(
			self, 
			c_start: CoordTup, c_end: CoordTup,
		) -> list[Coord]:
		"""find the shortest path between two coordinates, using A*"""

		

		g_score: dict[CoordTup, float] = dict() # cost of cheapest path to node from start currently known
		f_score: dict[CoordTup, float] = {c_start: 0.0} # estimated total cost of path thru a node: f_score[c] := g_score[c] + heuristic(c, c_end)

		# init
		g_score[c_start] = 0.0
		g_score[c_start] = self.heuristic(c_start, c_end)
	
		closed_vtx: set[CoordTup] = set() # nodes already evaluated
		open_vtx: set[CoordTup] = set([c_start]) # nodes to be evaluated
		source: dict[CoordTup, CoordTup] = dict() # node immediately preceding each node in the path (currently known shortest path)
	
		while open_vtx:
			# get lowest f_score node
			c_current: CoordTup = min(open_vtx, key=lambda c: f_score[c])
			# f_current: float = f_score[c_current]

			# check if goal is reached
			if c_end == c_current:
				path: list[CoordTup] = [c_current]
				p_current: CoordTup = c_current
				while p_current in source:
					p_current = source[p_current]
					path.append(p_current)
				return path[::-1]
	
			# close current node
			closed_vtx.add(c_current)
			open_vtx.remove(c_current)
	
			# update g_score of neighbors
			_np_neighbor: Coord
			for _np_neighbor in self.get_coord_neighbors(c_current):
				neighbor: CoordTup = tuple(_np_neighbor)

				if neighbor in closed_vtx:
					# already checked
					continue
				g_temp: float = g_score[c_current] + 1 # always 1 for maze neighbors

				if neighbor not in open_vtx:
					# found new vtx, so add
					open_vtx.add(neighbor)

				elif g_temp >= g_score[neighbor]:
					# if already knew about this one, but current g_score is worse, skip
					continue
	
				# store g_score and source
				source[neighbor] = c_current
				g_score[neighbor] = g_temp
				f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, c_end)


class LatticeMazeGenerators:
	"""namespace for generators for lattice mazes"""
	@staticmethod
	def gen_dfs(
			grid_shape: Coord, 
			start_coord: Coord|None = None,
			lattice_dim: int = 2,
		) -> "LatticeMaze":
		"""generate a lattice maze using depth first search, iterative
		
		algorithm:
		1. Choose the initial cell, mark it as visited and push it to the stack
		2. While the stack is not empty
			1. Pop a cell from the stack and make it a current cell
			2. If the current cell has any neighbours which have not been visited
				1. Push the current cell to the stack
				2. Choose one of the unvisited neighbours
				3. Remove the wall between the current cell and the chosen cell
				4. Mark the chosen cell as visited and push it to the stack
		"""
		# n_directions: int = lattice_dim * 2

		# initialize the maze with no connections)
		connection_list: NDArray[("lattice_dim", "x", "y"), bool] = np.zeros((lattice_dim, grid_shape[0], grid_shape[0]), dtype=bool)

		if start_coord is None:
			start_coord: Coord = (
				random.randint(0, grid_shape[0] - 1), 
				random.randint(0, grid_shape[1] - 1),
			)

		print(f"{grid_shape = } {start_coord = }")

		# initialize the stack with the target coord
		visited_cells: set[tuple[int, int]] = set(tuple(start_coord))
		stack: list[Coord] = [start_coord]

		# loop until the stack is empty
		while stack:
			# get the current coord from the stack
			current_coord: Coord = stack.pop()

			# filter neighbors by being within grid bounds and being unvisited
			unvisited_neighbors_deltas: list[tuple[Coord, Coord]] = [
				(neighbor, delta)
				for neighbor, delta in zip(current_coord + NEIGHBORS_MASK, NEIGHBORS_MASK)
				if (
					(tuple(neighbor) not in visited_cells)
					and (0 <= neighbor[0] < grid_shape[0]) 
					and (0 <= neighbor[1] < grid_shape[1])
				)
			]

			if unvisited_neighbors_deltas:
				stack.append(current_coord)

				# choose one of the unvisited neighbors
				chosen_neighbor, delta = random.choice(unvisited_neighbors_deltas)
				
				# add connection
				dim: int = np.argmax(np.abs(delta))
				# if positive, down/right from current coord
				# if negative, up/left from current coord (down/right from neighbor)
				clist_node: Coord = current_coord if (delta.sum() > 0) else chosen_neighbor
				connection_list[dim, clist_node[0], clist_node[1]] = True

				# add to visited cells and stack
				visited_cells.add(tuple(chosen_neighbor))
				stack.append(chosen_neighbor)

		return LatticeMaze(
			connection_list = connection_list,
			generation_meta = dict(
				func_name = "gen_dfs",
				grid_shape = grid_shape,
				start_coord = start_coord,
			),
		)

GENERATORS_MAP: dict[
	str, 
	Callable[[Coord, Any], "LatticeMaze"]
] = {
	"gen_dfs": LatticeMazeGenerators.gen_dfs,
}



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


			

















def _test(shape_x: int = 5, shape_y: int|None = None):
	if shape_y is None:
		shape_y = shape_x

	t = time.time()
	m = LatticeMazeGenerators.gen_dfs(np.array([shape_x, shape_y]))
	print(f"generation time: {time.time() - t}")
	# print(m)
	# show the maze
	img = m.as_img()

	# show a path
	c_start = (0, 0)
	c_end = (shape_x - 1, shape_y - 1)
	
	t = time.time()
	path = m.find_shortest_path(
		c_start = c_start,
		c_end = c_end,
	)
	print(f"solving time: {time.time() - t}")
	path = m.points_transform_to_img(np.array(path))
	plt.plot(*zip(*path), "-", color="red")
	plt.plot([path[0][0]], [path[0][1]], "o", color="red")
	plt.plot([path[-1][0]], [path[-1][1]], "x", color="red")
	plt.imshow(img.T, cmap="gray")
	plt.show()

if __name__ == "__main__":
	import fire
	fire.Fire(_test)