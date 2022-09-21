from functools import partial
from pathlib import Path
import typing
import multiprocessing
import os

import numpy as np
from tqdm import tqdm
from muutils.misc import shorten_numerical_to_str

from maze_transformer.generation.latticemaze import LatticeMaze, Coord, CoordArray, CoordTup
from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.training.mazedataset import MazeDatasetConfig, MazeDataset
from maze_transformer.training.tokenizer import MazeTokenizer


def generate_MazeTokenizer(
		junk,
		grid_n: int,
		c_start: tuple[int, int],
		c_end: tuple[int, int],
	) -> MazeTokenizer:
	
	maze = LatticeMazeGenerators.gen_dfs(
		grid_shape=(grid_n, grid_n),
		lattice_dim=2,
	)
	return MazeTokenizer(
		maze=maze,
		solution=np.array(maze.find_shortest_path(
			c_start=c_start,
			c_end=c_end,
		)),
	)


def create(
		path_base: str,
		n_mazes: int,
		grid_n: int = 16,
		name: str|None = None,
		**cfg_kwargs,
	):

	name_base: str = f"g{grid_n}-n{shorten_numerical_to_str(n_mazes, small_as_decimal = False)}"
	if name is None:
		name = name_base
	else:
		name = f"{name_base}-{name}"

	data_path: str = Path(path_base) / name
	print(f"generating dataset: {data_path.as_posix() = } {n_mazes = } {grid_n = } {name = } {cfg_kwargs = }")

	if os.path.exists(data_path):
		raise FileExistsError(f"path {data_path} already exists!")

	# create config
	# TODO: figure out unexpected keyword argument linter error here?
	cfg: MazeDatasetConfig = MazeDatasetConfig(
		name = name,
		grid_n = grid_n,
		n_mazes = n_mazes,
		**cfg_kwargs,
	)

	# create and solve mazes
	c_start = (0, 0)
	c_end = (cfg.grid_n - 1, cfg.grid_n - 1)

	mazes: list[MazeTokenizer] 
	
	with multiprocessing.Pool() as pool:
		mazes = list(tqdm(
			pool.imap(
				partial(generate_MazeTokenizer, grid_n=grid_n, c_start=c_start, c_end=c_end),
				range(cfg.n_mazes),
			),
			total = cfg.n_mazes,
			unit = "maze",
			desc="generating & solving mazes",			
		))

	# create and save dataset
	dataset: MazeDataset = MazeDataset(
		cfg=cfg,
		mazes_objs=mazes,
	)

	dataset.disk_save(data_path)


def load(path: str) -> None:
	d = MazeDataset.disk_load(path)

	print(d.cfg)
	print(d.mazes_array)

	print("done!")


if __name__ == "__main__":
	import fire
	fire.Fire(dict(
		create=create,
		load=load,
	))



