from functools import partial
import typing
import multiprocessing

import numpy as np
from tqdm import tqdm

from maze_transformer.latticemaze import LatticeMaze, Coord, CoordArray, CoordTup
from maze_transformer.generators import LatticeMazeGenerators
from maze_transformer.tokenizer import MazeDatasetConfig, MazeDataset, SolvedMaze


def generate_solvedmaze(
		junk,
		grid_n: int,
		c_start: tuple[int, int],
		c_end: tuple[int, int],
	) -> SolvedMaze:
	
	maze = LatticeMazeGenerators.gen_dfs(
		grid_shape=(grid_n, grid_n),
		lattice_dim=2,
	)
	return SolvedMaze(
		maze=maze,
		solution=np.array(maze.find_shortest_path(
			c_start=c_start,
			c_end=c_end,
		)),
	)


def create(
		path_base: str = "data/test-001/data",
		n_mazes: int = 64,
		grid_n: int = 16,
		**cfg_kwargs,
	):

	# create config
	cfg: MazeDatasetConfig = MazeDatasetConfig(
		grid_n=grid_n,
		n_mazes=n_mazes,
		**cfg_kwargs,
	)

	# create and solve mazes
	c_start = (0, 0)
	c_end = (cfg.grid_n - 1, cfg.grid_n - 1)

	mazes: list[SolvedMaze] 
	
	with multiprocessing.Pool() as pool:
		mazes = list(tqdm(
			pool.imap(
				partial(generate_solvedmaze, grid_n=grid_n, c_start=c_start, c_end=c_end),
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

	dataset.disk_save(path_base = path_base)


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



