import json
import typing
from pathlib import Path

import torch
from muutils.tensor_utils import NDArray
from transformer_lens import HookedTransformer

from maze_transformer.generation.latticemaze import (
    SPECIAL_TOKENS,
    CoordTup,
    LatticeMaze,
)
from maze_transformer.training.config import ConfigHolder
from maze_transformer.training.mazedataset import MazeDatasetConfig
from maze_transformer.training.tokenizer import SPECIAL_TOKENS
from maze_transformer.training.training import TRAIN_SAVE_FILES

# pylint: disable=protected-access

MazePath = list[CoordTup]
ArrMazePath = NDArray["node x_y_pos", int]


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

    # TODO: replace this whole thing with a single zanj.read(fname) call
    """
    # load the configs
    # check for the filenames, go up a dir if they don't exist
    assert model_path.suffix == ".pt", "Model path must be a .pt file"
    config_path = find_config(model_path)

    assert (
        config_path is not None
    ), f"Couldn't find configs in run containing {model_path}"

    # TODO Make this part of the ConfigHolder - https://github.com/AISC-understanding-search/maze-transformer/issues/31

    # load the configs
    with open(config_path, "r") as f:
        combined_json = json.load(f)
        config_holder = ConfigHolder.load(combined_json)

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
    # NOTE temporary fix until https://github.com/neelnanda-io/TransformerLens/issues/219 is resolved

    model.process_weights_(fold_ln=True)
    model.setup()  # Re-attach layernorm hooks by calling setup
    model.eval()

    return model, config_holder


def decode_maze_tokens_to_coords(
    tokens: list[str],
    mazedata_cfg: MazeDatasetConfig,
    when_noncoord: typing.Literal["except", "skip", "include"] = "skip",
) -> list[str] | list[tuple[int, int]]:
    """given a list of tokens, decode the coordinate-tokens to a list of coordinates, leaving other tokens as-is"""
    output: list[str] | list[tuple[int, int]] = list()
    for idx, tk in enumerate(tokens):
        if tk in mazedata_cfg.token_node_map:
            output.append(mazedata_cfg.token_node_map[tk])
        else:
            if when_noncoord == "skip":
                continue
            elif when_noncoord == "include":
                output.append(tk)
            elif when_noncoord == "except":
                raise ValueError(f"token '{tk}' at {i = } is not a coordinate")
            else:
                raise ValueError(f"invalid value for {when_noncoord = }")
    return output


def predict_maze_path(
    tokens: list[str],
    data_cfg: MazeDatasetConfig,
    model: HookedTransformer,
    include_start_coord: bool = True,
    n_tokens_pred: int = 8,
    verbose: bool = False,
) -> tuple[LatticeMaze, MazePath, MazePath]:
    """given tokens from a dataset, predict the next tokens with the model, decode both true and predicted to paths

    ### Parameters:
     - `tokens : list[str]`
       raw tokens from dataset, containing both maze and true path
     - `data_cfg : MazeDatasetConfig`
       config for the dataset
     - `model : HookedTransformer`
       model to use for prediction
     - `n_tokens_pred : int`
       number of tokens to predict
     - `**generate_kwargs`
       additional keyword arguments to pass to `model.generate`

    ### Returns:
     - `tuple[MazePath, MazePath]`
       ( path_true, path_predicted )
    """

    # split the tokens into maze (prompt) and path
    path_start_token: str = SPECIAL_TOKENS["path_start"]
    path_start_index: int = tokens.index(path_start_token) + 1
    maze_tokens: list[str] = tokens[:path_start_index]
    path_true_tokens: list[str] = tokens[path_start_index:]

    if include_start_coord:
        # add the first coordinate to `maze_tokens`
        maze_tokens.append(path_true_tokens[0])

    # encode + pad the maze tokens
    maze_arr_nopad = model.to_tokens(" ".join(maze_tokens), prepend_bos=False)

    if verbose:
        print("Generating Model Completions")
    #! NOTE verbose flag here will require latest clone of TrasformerLens from github
    # have the model predict some tokens
    predictions = model.generate(
        maze_arr_nopad,
        eos_token_id=model.tokenizer.eos_token_id,
        stop_at_eos=True,
        max_new_tokens=n_tokens_pred,
        verbose=verbose,
    )

    # decode the tokens
    predicted_and_context_tokens = model.to_str_tokens(predictions)
    pac_path_start_idx: int = predicted_and_context_tokens.index(path_start_token) + 1
    predicted_tokens: list[str] = predicted_and_context_tokens[pac_path_start_idx:]

    if verbose:
        print(
            f"{maze_tokens = }\n{path_true_tokens = }\n{predicted_and_context_tokens = }\n{predicted_tokens = }"
        )

    # convert tokens to coordinates
    path_true = decode_maze_tokens_to_coords(
        path_true_tokens,
        mazedata_cfg=data_cfg,
        when_noncoord="skip",
    )

    path_predicted = decode_maze_tokens_to_coords(
        predicted_tokens,
        mazedata_cfg=data_cfg,
        when_noncoord="skip",
    )

    # remove start and end tokens from maze_tokens
    maze_tokens = maze_tokens[
        maze_tokens.index(SPECIAL_TOKENS["adjlist_start"])
        + 1 : maze_tokens.index(SPECIAL_TOKENS["adjlist_end"])
    ]

    return (
        LatticeMaze.from_tokens(maze_tokens),
        path_true,
        path_predicted,
    )
