from typing import Any, Callable, Generic, Literal, NamedTuple, Sequence, TypeVar, Union

import numpy as np
import matplotlib.pyplot as plt
from muutils.tensor_utils import ATensor, NDArray, DTYPE_MAP

from maze_transformer.latticemaze import LatticeMaze

def plot_path(maze: LatticeMaze, path: NDArray, show: bool = True) -> None:
	# print(m)
	# show the maze
	img = maze.as_img()
	path_transformed = maze.points_transform_to_img(path)

	# plot path
	plt.plot(*zip(*path_transformed), "-", color="red")
	# mark endpoints
	plt.plot([path_transformed[0][0]], [path_transformed[0][1]], "o", color="red")
	plt.plot([path_transformed[-1][0]], [path_transformed[-1][1]], "x", color="red")
	# show actual maze
	plt.imshow(img.T, cmap="gray")

	if show:
		plt.show()
