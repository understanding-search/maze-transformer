import multiprocessing
import os
from pathlib import Path
import warnings

from muutils.misc import shorten_numerical_to_str  # type: ignore[import]
from tqdm import tqdm

from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.training.maze_dataset import MazeDataset, MazeDatasetConfig


def create_dataset(
    path_base: str,
    n_mazes: int,
    grid_n: int = 16,
    name: str | None = None,
    verbose: bool = False,
    **cfg_kwargs,
):
    warnings.warn(
        "create_dataset is deprecated, use MazeDataset.from_config instead",
        DeprecationWarning,
    )
    if n_mazes < 0:
        raise ValueError("n_mazes must be >= 0")
    if grid_n < 0:
        raise ValueError("grid_n must be >= 0")

    name_base: str = (
        f"g{grid_n}-n{shorten_numerical_to_str(n_mazes, small_as_decimal = False)}"
    )
    if name is None:
        name = name_base
    else:
        name = f"{name_base}-{name}"

    data_path = Path(path_base) / name
    if verbose:
        print(
            f"generating dataset: {data_path.as_posix() = } {n_mazes = } {grid_n = } {name = } {cfg_kwargs = }"
        )

    if os.path.exists(data_path):
        raise FileExistsError(f"path {data_path} already exists!")

    # create config
    config: MazeDatasetConfig = MazeDatasetConfig(
        name=name,
        grid_n=grid_n,
        n_mazes=n_mazes,
        **cfg_kwargs,
    )

    # create and solve mazes
    with multiprocessing.Pool() as pool:
        solved_mazes = list(
            tqdm(
                pool.imap(
                    LatticeMazeGenerators.gen_dfs_with_solution,
                    ((grid_n, grid_n) for _ in range(config.n_mazes)),
                ),
                total=config.n_mazes,
                unit="maze",
                desc="generating & solving mazes",
                disable=not verbose,
            )
        )

    # create and save dataset
    dataset: MazeDataset = MazeDataset(
        cfg=config,
        mazes=solved_mazes,
    )

    dataset.save(str(data_path))


def load(path: str) -> None:
    d = MazeDataset.disk_load(path, do_tokenized=True)

    print(d.cfg)
    print(d.mazes_array)

    print("done!")


if __name__ == "__main__":
    import fire

    fire.Fire(
        dict(
            create=create_dataset,
            load=load,
        )
    )
