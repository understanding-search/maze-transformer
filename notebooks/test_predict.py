# Generic
import typing
import os
from pathlib import Path
import cProfile
import re

# Plotting
import matplotlib.pyplot as plt

# Numerical Computing
import numpy as np
import torch

# Utilities
from muutils.statcounter import StatCounter

# Our Code
from maze_transformer.utils.utils import set_reproducibility
from maze_transformer.generation.latticemaze import LatticeMaze
from maze_transformer.evaluation.plot_maze import plot_multi_paths, PathFormat
from maze_transformer.evaluation.eval_model import (
    MazePath,
    ArrMazePath,
    load_model_with_configs,
    predict_maze_path,
)
from maze_transformer.evaluation.pathdist import (
    MazeEvalFunction,
    ArrMazeEvalFunction,
    MazeEvalFuncs,
    ArrMazeEvalFuncs,
)

# Setup
# device = configure_notebook(seed=42, dark_mode=True)
# We won't be training any models
torch.set_grad_enabled(False)
set_reproducibility()

# Get latest model
# this should point towards a directory containing a run.
# If you don't have any runs, you can create a dataset with `poetry run python scripts/create_dataset.py create ./data/maze 10 --grid_n=4`
# Then train a model with poetry run python scripts/train_model.py ./data/maze/g4-n10`
run_path = Path("./data/maze/g4-n10")
assert run_path.exists(), f"Run path {run_path.as_posix()} does not exist"
model_path = list(sorted(run_path.glob("**/model.final.pt"), key=os.path.getmtime))[
    -1
].resolve()
maze_path = run_path / "maze_tokens.jsonl"

EvalFuncTuple = tuple[
    typing.Literal["arr", "list"], MazeEvalFunction | ArrMazeEvalFunction
]

ALL_PATHDIST_FUNCS: dict[str, EvalFuncTuple] = {
    **{
        name: ("arr", func)
        for name, func in ArrMazeEvalFuncs.__dict__.items()
        if not name.startswith("_")
    },
    **{
        name: ("list", func)
        for name, func in MazeEvalFuncs.__dict__.items()
        if not name.startswith("_")
    },
}

print(ALL_PATHDIST_FUNCS)

from datetime import datetime


def evaluate_model_pathdist_scores(
    model_path: Path,
    maze_tokens_path: Path,
    pathdist_functions: dict[str, EvalFuncTuple] | None = ALL_PATHDIST_FUNCS,
    n_tokens_pred: int = 8,
    n_mazes: int = 64,
    verbose: bool = False,
) -> dict[str, StatCounter]:

    mazes_tokens: list[list[str]] = list()

    # load model and configs
    model, cfg = load_model_with_configs(model_path)

    # load maze test data
    mazes_tokens: list[list[str]] = [
        line.split() for line in maze_tokens_path.read_text().splitlines()
    ]

    mazes_tokens_initial = mazes_tokens
    mazes_tokens_after = []
    with open(maze_tokens_path, "r") as f:
        for idx, line in enumerate(f):
            mazes_tokens_after.append(line.split())
            if idx >= n_mazes:
                break
    assert all([x == y for x, y in zip(mazes_tokens_initial, mazes_tokens_after)])
    print("passed check")

    # predict paths
    mazes_solved: list[tuple[LatticeMaze, MazePath, MazePath]] = list()
    start = datetime.now()
    # This is 99% of the runtime
    for tokens in mazes_tokens:
        maze, p_true, p_pred = predict_maze_path(
            tokens=tokens,
            data_cfg=cfg.dataset_cfg,
            model=model,
            n_tokens_pred=n_tokens_pred,
            verbose=verbose,
        )

        mazes_solved.append((maze, p_true, p_pred))

        if verbose:
            print(f"{p_true = }")
            print(f"{p_pred = }")

    print(mazes_solved[0])

    print("predicting time:", datetime.now() - start)
    # convert paths
    mazes_solved_arrpath: list[tuple[LatticeMaze, ArrMazePath, ArrMazePath]] = [
        (maze, np.array(p_true), np.array(p_pred))
        for maze, p_true, p_pred in mazes_solved
    ]

    # evaluate
    pathdist_scores: dict[str, StatCounter] = dict()
    for name, (pathdist_type, pathdist_func) in pathdist_functions.items():
        if pathdist_type == "list":
            pathdist_scores[name] = StatCounter(
                pathdist_func(maze, p_true, p_pred)
                for maze, p_true, p_pred in mazes_solved
            )
        elif pathdist_type == "arr":
            pathdist_scores[name] = StatCounter(
                pathdist_func(maze, p_true, p_pred)
                for maze, p_true, p_pred in mazes_solved_arrpath
            )
        else:
            raise ValueError(f"Invalid pathdist_type: {pathdist_type}")

    return pathdist_scores


evaluate_model_pathdist_scores(model_path, maze_path)
