# Generic
import os
import typing
from pathlib import Path

# Numerical Computing
import numpy as np
import torch

# Utilities
from muutils.statcounter import StatCounter

from maze_transformer.evaluation.eval_model import (
    ArrMazePath,
    MazePath,
    load_model_with_configs,
    predict_maze_path,
    predict_maze_paths,
)
from maze_transformer.evaluation.pathdist import (
    ArrMazeEvalFuncs,
    ArrMazeEvalFunction,
    MazeEvalFuncs,
    MazeEvalFunction,
)
from maze_transformer.generation.latticemaze import LatticeMaze, SolvedMaze

# Our Code
from maze_transformer.utils.utils import chunks, set_reproducibility

# Plotting


# Setup
# device = configure_notebook(seed=42, dark_mode=True)
# We won't be training any models
torch.set_grad_enabled(False)
set_reproducibility()

# Get latest model
# this should point towards a directory containing a run.
# If you don't have any runs, you can create a dataset with `poetry run python scripts/create_dataset.py create ./data/maze 10 --grid_n=4`
# Then train a model with poetry run python scripts/train_model.py ./data/maze/g4-n10`
run_path = Path("./data/maze/g4-n100")
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


def evaluate_model_pathdist_scores(
    model_path: Path,
    maze_tokens_path: Path,
    pathdist_functions: dict[str, EvalFuncTuple] | None = ALL_PATHDIST_FUNCS,
    n_tokens_pred: int = 8,
    n_mazes: int = 64,
    verbose: bool = False,
) -> dict[str, StatCounter]:
    # load model and configs
    model, cfg = load_model_with_configs(model_path)

    # load maze test data
    mazes_tokens: list[list[str]] = [
        line.split() for line in maze_tokens_path.read_text().splitlines()
    ]

    # predict paths
    mazes_solved: list[tuple[LatticeMaze, MazePath, MazePath]] = list()
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

    # convert paths
    mazes_solved_arrpath: list[tuple[LatticeMaze, ArrMazePath, ArrMazePath]] = [
        (maze, np.array(p_true), np.array(p_pred))
        for maze, p_true, p_pred in mazes_solved
    ]
    return mazes_solved_arrpath

    # evaluate
    # pathdist_scores: dict[str, StatCounter] = dict()
    # for name, (pathdist_type, pathdist_func) in pathdist_functions.items():
    #     # TODO: notebook should not worry about lists vs np arrays - either standardize all eval functions on one, or have them all accept both and convert as needed
    #     # alternative start using np arrays everywhere
    #     if pathdist_type == "list":
    #         pathdist_scores[name] = StatCounter(
    #             pathdist_func(maze, p_true, p_pred)
    #             for maze, p_true, p_pred in mazes_solved
    #         )
    #     elif pathdist_type == "arr":
    #         pathdist_scores[name] = StatCounter(
    #             pathdist_func(maze, p_true, p_pred)
    #             for maze, p_true, p_pred in mazes_solved_arrpath
    #         )
    #     else:
    #         raise ValueError(f"Invalid pathdist_type: {pathdist_type}")

    # return pathdist_scores


def evaluate_model_pathdist_scores2(
    model_path: Path,
    maze_tokens_path: Path,
    pathdist_functions: dict[str, EvalFuncTuple] | None = ALL_PATHDIST_FUNCS,
    n_tokens_pred: int = 8,
    batch_size: int = 64,
    verbose: bool = False,
) -> dict[str, StatCounter]:
    model, cfg = load_model_with_configs(model_path)

    # TODO: This seems like a utility fn - where to put it?
    tokenized_mazes: list[list[str]] = [
        line.split() for line in maze_tokens_path.read_text().splitlines()
    ]

    solved_mazes = [
        SolvedMaze.from_tokens(tokens, cfg.dataset_cfg) for tokens in tokenized_mazes
    ]
    predicted_paths: list[list[tuple[int, int]]] = []
    for batch in chunks(tokenized_mazes, batch_size):
        predicted_paths += predict_maze_paths(
            tokens_batch=batch,
            data_cfg=cfg.dataset_cfg,
            model=model,
            n_tokens_pred=n_tokens_pred,
        )

    return predicted_paths
    # return pathdist_scores


from datetime import datetime

one_start = datetime.now()
one = evaluate_model_pathdist_scores(model_path, maze_path)
one_end = datetime.now()
two = evaluate_model_pathdist_scores2(model_path, maze_path)
two_end = datetime.now()
print("one - ", one_end - one_start)
print("two - ", two_end - one_end)
