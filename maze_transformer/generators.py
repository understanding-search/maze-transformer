import sys
import random
from typing import Any, Callable, Generic, Literal, NamedTuple, Sequence, TypeVar, Union

import numpy as np
from muutils.tensor_utils import ATensor, NDArray, DTYPE_MAP
from muutils.json_serialize import json_serialize, dataclass_serializer_factory, dataclass_loader_factory, try_catch, JSONitem

from maze_transformer.latticemaze import LatticeMaze, Coord, CoordArray, CoordTup, NEIGHBORS_MASK

class LatticeMazeGenerators:
	"""namespace for lattice maze generation algorithms"""
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
		connection_list: np.ndarray = np.zeros((lattice_dim, grid_shape[0], grid_shape[0]), dtype=bool)

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