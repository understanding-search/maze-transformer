import multiprocessing
import os
from functools import partial
from pathlib import Path

import numpy as np
from muutils.misc import shorten_numerical_to_str  # type: ignore[import]
from tqdm import tqdm

from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.training.mazedataset import MazeDataset, MazeDatasetConfig
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
        solution=np.array(
            maze.find_shortest_path(
                c_start=c_start,
                c_end=c_end,
            )
        ),
    )


def create(
    path_base: str,
    n_mazes: int,
    grid_n: int = 16,
    name: str | None = None,
    **cfg_kwargs,
):
    if n_mazes < 0:
        raise ValueError("m_mazes must be >= 0")

    name_base: str = (
        f"g{grid_n}-n{shorten_numerical_to_str(n_mazes, small_as_decimal = False)}"
    )
    if name is None:
        name = name_base
    else:
        name = f"{name_base}-{name}"

    data_path = Path(path_base) / name
    print(
        f"generating dataset: {data_path.as_posix() = } {n_mazes = } {grid_n = } {name = } {cfg_kwargs = }"
    )

    if os.path.exists(data_path):
        raise FileExistsError(f"path {data_path} already exists!")

    # create config
    # TODO: figure out unexpected keyword argument linter error here?
    config: MazeDatasetConfig = MazeDatasetConfig(
        name=name,
        grid_n=grid_n,
        n_mazes=n_mazes,
        **cfg_kwargs,
    )

    # create and solve mazes
    top_left = (0, 0)
    bottom_right = (config.grid_n - 1, config.grid_n - 1)

    mazes: list[MazeTokenizer]

    with multiprocessing.Pool() as pool:
        mazes = list(
            tqdm(
                pool.imap(
                    partial(
                        generate_MazeTokenizer,
                        grid_n=grid_n,
                        c_start=top_left,
                        c_end=bottom_right,
                    ),
                    range(config.n_mazes),
                ),
                total=config.n_mazes,
                unit="maze",
                desc="generating & solving mazes",
            )
        )

    # create and save dataset
    dataset: MazeDataset = MazeDataset(
        cfg=config,
        mazes_objs=mazes,
    )

    dataset.disk_save(str(data_path))


def load(path: str) -> None:
    d = MazeDataset.disk_load(path, do_tokenized=True)

    print(d.cfg)
    print(d.mazes_array)

    print("done!")


if __name__ == "__main__":
    import fire # type: ignore[import]

    fire.Fire(
        dict(
            create=create,
            load=load,
        )
    )
