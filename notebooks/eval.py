import os
from pathlib import Path

import numpy as np
import torch

from muutils.statcounter import StatCounter

from maze_transformer.evaluation.eval_model import (
    evaluate_model,
    load_model_with_configs,
)
from maze_transformer.evaluation.path_evals import PathEvals, PathEvalFunction, MazePath
from maze_transformer.generation.latticemaze import LatticeMaze

from maze_transformer.utils.utils import set_reproducibility

torch.set_grad_enabled(False)
set_reproducibility()

run_path = Path("./data/maze/g4-n100")
assert run_path.exists(), f"Run path {run_path.as_posix()} does not exist"
model_path = list(sorted(run_path.glob("**/model.final.pt"), key=os.path.getmtime))[
    -1
].resolve()
maze_path = run_path / "maze_tokens.jsonl"


def evaluate_pathdist_scores_checkpoints(
    run_path: Path,  # Path to run, not model.final.pt or checkpoints
    maze_tokens_path: Path,
    checkpoint_idxs: list[int] | None = None,
    # pathdist_functions: dict[str, EvalFuncTuple]|None = ALL_PATHDIST_FUNCS,
    # skip_every_nth: int = 1,
    # n_tokens_pred: int = 8,
    # n_mazes: int = 10,
    # verbose: bool = False,
) -> dict[str, dict[int, StatCounter]]:

    model_checkpoints: list[tuple[int, Path]]
    assert (
        run_path.is_dir()
    ), f"Model path {run_path} is not a directory (expect run directory, not model files)"

    # this is utility - loading checkpoints from a run dir
    if checkpoint_idxs is not None:
        model_checkpoints = list()
        for idx in checkpoint_idxs:
            mdl_path: Path = Path(run_path) / f"checkpoints/model.iter_{idx}.pt"
            if not mdl_path.exists():
                raise ValueError(f"Checkpoint file {mdl_path} does not exist")
            model_checkpoints.append((idx, mdl_path))
    else:
        model_checkpoints = [
            (int(mdl_path.stem.split("_")[-1].split(".")[0]), mdl_path)
            for mdl_path in sorted(Path(run_path).glob("checkpoints/model.iter_*.pt"))
        ]

    print(f"Found {len(model_checkpoints)} checkpoints:\n\t{model_checkpoints = }")

    pathdist_scores_idx: dict[int, dict[str, StatCounter]] = dict()

    # skip not needed... utility returns list of checkpoints, just slice that in notebook
    for idx, mdl_path in model_checkpoints:

        print(f"# Evaluating checkpoint {idx} at {mdl_path}")
        pathdist_scores_idx[idx] = evaluate_model(
            model_path=mdl_path,
            # pathdist_functions = pathdist_functions,
            # n_tokens_pred = n_tokens_pred,
            maze_tokens_path=maze_tokens_path,
            # n_mazes = n_mazes,
            # verbose = verbose,
        )

    return {
        name: {idx: scores[name] for idx, scores in pathdist_scores_idx.items()}
        for name in pathdist_scores_idx[0]
    }


data = evaluate_pathdist_scores_checkpoints(
    run_path=model_path.parent,
    maze_tokens_path=maze_path,
    # n_mazes = 4,
    # skip_every_nth=10,
    # verbose = True,
)

breakpoint()
