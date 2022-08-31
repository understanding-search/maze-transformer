import time
from typing import Any, Callable, Generic, Literal, NamedTuple, Sequence, TypeVar, Union

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from muutils.tensor_utils import ATensor, NDArray, DTYPE_MAP


from maze_transformer.latticemaze import LatticeMaze
from maze_transformer.generators import LatticeMazeGenerators
from maze_transformer.plot import plot_path

	
def generate_solve_plot(shape_x: int = 5, shape_y: int|None = None):
	if shape_y is None:
		shape_y = shape_x

	t: float = time.time()
	m: LatticeMaze = LatticeMazeGenerators.gen_dfs(np.array([shape_x, shape_y]))
	print(f"generation time: {time.time() - t}")

	# show a path
	c_start = (0, 0)
	c_end = (shape_x - 1, shape_y - 1)
	
	t = time.time()
	path = np.array(m.find_shortest_path(
		c_start = c_start,
		c_end = c_end,
	))

	print(f"solving time: {time.time() - t}")

	plot_path(m, path, show=True)
	
	

if __name__ == "__main__":
	import fire
	fire.Fire(generate_solve_plot)