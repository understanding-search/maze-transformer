import typing

import numpy as np
from tqdm import tqdm

from maze_transformer.latticemaze import LatticeMaze, Coord, CoordArray, CoordTup
from maze_transformer.generators import LatticeMazeGenerators
from maze_transformer.tokenizer import MazeDatasetConfig, MazeDataset, SolvedMaze


def create(
		path_base: str = "data/test-001/data",
		grid_n: int = 16,
		n_mazes: int = 64,
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

	mazes: list[SolvedMaze] = list()
	for _ in tqdm(
			range(cfg.n_mazes),
			total=cfg.n_mazes,
			desc="generating & solving mazes",			
		):
		maze = LatticeMazeGenerators.gen_dfs(
			grid_shape=(cfg.grid_n, cfg.grid_n),
			lattice_dim=2,
		)
		mazes.append(
			SolvedMaze(
				maze=maze,
				solution=np.array(maze.find_shortest_path(
					c_start=c_start,
					c_end=c_end,
				)),
			)
		)

	# create and save dataset
	dataset: MazeDataset = MazeDataset(
		cfg=cfg,
		mazes_objs=mazes,
	)

	dataset.disk_save(path_base = path_base)


def load(path: str) -> None:
	d = MazeDataset.disk_load(path)

	print(d.cfg)
	print(d.mazes_tokenized)

	print("done!")


if __name__ == "__main__":
	import fire
	fire.Fire(dict(
		create=create,
		load=load,
	))



