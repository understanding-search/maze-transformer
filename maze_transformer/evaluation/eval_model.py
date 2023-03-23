import json

# Generic
from pathlib import Path
from typing import cast

# Numerical Computing
import numpy as np
import torch

# Utilities
from muutils.statcounter import StatCounter
from muutils.tensor_utils import ATensor
from transformer_lens import HookedTransformer

# bin these
from transformers import PreTrainedTokenizer

from maze_transformer.evaluation.path_evals import PathEvalFunction, PathEvals
from maze_transformer.generation.constants import SPECIAL_TOKENS
from maze_transformer.generation.latticemaze import SolvedMaze
from maze_transformer.training.config import ConfigHolder
from maze_transformer.training.mazedataset import MazeDatasetConfig
from maze_transformer.training.training import TRAIN_SAVE_FILES
from maze_transformer.utils.token_utils import (
    decode_maze_tokens_to_coords,
    get_path_tokens,
    get_tokens_up_to_path_start,
)

# Our Code
from maze_transformer.utils.utils import chunks

# pylint: disable=protected-access


def find_config(folder: Path) -> Path | tuple[Path, Path] | None:
    """Assumed directory structure:
    run_folder/
        -- model.final.pt
        -- config.json (or data_config.json and train_config.json)
        -- checkpoints/
            -- model.checkpoint_num.pt

    Should be able to return config from anywhere in this structure
    (regardless if path provided is file or folder name)
    """
    containing_folder = folder if folder.is_dir() else folder.parent
    if containing_folder.name == "checkpoints":
        to_check = [containing_folder.parent.parent]  # get run folder from checkpoints
    else:  # Generic - probably got path to run folder or model.final.pt
        to_check = [containing_folder, containing_folder.parent]

    for folder in to_check:
        holder_path = folder / TRAIN_SAVE_FILES.config_holder
        if holder_path.exists():
            return holder_path


def load_model_with_configs(
    model_path: Path,
    verbose: bool = False,
) -> tuple[HookedTransformer, ConfigHolder]:
    """
    Load a model and associated config files from a path.
    """

    # Note: This should become less fragile in the future when we start keeping configs in the dataset file

    # get path to the folder containing the model
    config_folder: Path = model_path.parent
    # check for the filenames, go up a dir if they don't exist
    assert model_path.suffix == ".pt", "Model path must be a .pt file"
    config_path = find_config(model_path)

    assert (
        config_path is not None
    ), f"Couldn't find configs in run containing {model_path}"

    # TODO Make this part of the ConfigHolder - https://github.com/AISC-understanding-search/maze-transformer/issues/31
    # initialize tokenizer
    tokenizer = PreTrainedTokenizer(
        bos_token=SPECIAL_TOKENS["padding"],
        eos_token=SPECIAL_TOKENS["padding"],
        pad_token=SPECIAL_TOKENS["padding"],
    )

    # load the configs
    with open(config_path, "r") as f:
        combined_json = json.load(f)
        config_holder = ConfigHolder.load(combined_json)
        config_holder.tokenizer = tokenizer

    if verbose:
        print(f"Loaded config\n{config_holder}\n" + ("-" * 40))

    model: HookedTransformer = config_holder.create_model()
    state_dict = torch.load(model_path, map_location=model.cfg.device)
    model.load_and_process_state_dict(
        state_dict,
        fold_ln=False,
        center_writing_weights=True,
        center_unembed=True,
        refactor_factored_attn_matrices=True,
    )
    # We're folding layernorm, but not using HookedTransformer.from_pretrained
    # This means when torch.load_state_dict is invoked by transformer_lens, it
    # will complain about the fact that we deleted layernorm from the state_dict
    # Neel has a function to address this which Inserts layernorms that implement
    # only the normalization - to not trigger warning from torch.load - disgusting so:
    # TODO Probably just get Neel to allow the "strict" flag to be disabled inside of model.load_and_process_state_dict
    model.process_weights_(fold_ln=True)
    model.eval()

    return model, config_holder


def predict_maze_paths(
    tokens_batch: list[list[str]],
    data_cfg: MazeDatasetConfig,
    model: HookedTransformer,
    n_tokens_pred: int = 8,
    verbose: bool = False,
) -> list[list[tuple[int, int]]]:
    indices_batch: list[ATensor] = []
    for tokens in tokens_batch:
        maze = get_tokens_up_to_path_start(tokens)
        indices = torch.tensor(
            [data_cfg.tokenizer_map[t] for t in maze], dtype=torch.long
        )
        indices_batch.append(indices)

    prediction_batch = model.generate(
        torch.stack(indices_batch),
        eos_token_id=data_cfg.tokenizer_map[SPECIAL_TOKENS["path_end"]],
        stop_at_eos=True,
        max_new_tokens=n_tokens_pred,
        verbose=verbose,
    )

    paths: list[list[tuple[int, int]]] = []
    for preds in prediction_batch:
        pred_tokens: list[str] = [data_cfg.token_arr[t] for t in preds]
        path_tokens = get_path_tokens(pred_tokens)
        path_coords = decode_maze_tokens_to_coords(
            path_tokens, mazedata_cfg=data_cfg, when_noncoord="skip"
        )
        # This is the correct type when using "skip"
        paths.append(cast(list[tuple[int, int]], path_coords))

    return paths


def evaluate_model(
    model_path: Path,
    maze_tokens_path: Path,
    eval_functions: dict[str, PathEvalFunction] | None = None,
    n_tokens_pred: int = 8,
    batch_size: int = 64,
    verbose: bool = False,
) -> dict[str, StatCounter]:
    if not eval_functions:
        eval_functions = PathEvals.all_functions()

    model, cfg = load_model_with_configs(model_path)
    score_counters: dict[str, StatCounter] = {
        name: StatCounter() for name in eval_functions.keys()
    }

    with maze_tokens_path.open() as token_file:
        for batch in chunks(token_file, batch_size):
            tokenized_mazes = [line.split() for line in batch]

            solved_mazes = [
                SolvedMaze.from_tokens(tokens, cfg.dataset_cfg)
                for tokens in tokenized_mazes
            ]
            mazes, solutions = zip(*solved_mazes)

            predictions = predict_maze_paths(
                tokens_batch=tokenized_mazes,
                data_cfg=cfg.dataset_cfg,
                model=model,
                n_tokens_pred=n_tokens_pred,
            )

            for name, func in eval_functions.items():
                score_counters[name].update(
                    func(maze, np.array(solution), np.array(prediction))
                    for maze, solution, prediction in zip(mazes, solutions, predictions)
                )

    return score_counters
