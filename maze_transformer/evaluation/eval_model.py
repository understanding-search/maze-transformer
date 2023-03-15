import json
import typing
from pathlib import Path

import numpy as np
import torch
from muutils.tensor_utils import ATensor, NDArray
from transformer_lens import HookedTransformer

# bin these
from transformers import PreTrainedTokenizer

from maze_transformer.evaluation.plot_maze import PathFormat, plot_multi_paths
from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.generation.latticemaze import (
    SPECIAL_TOKENS,
    CoordTup,
    LatticeMaze,
)
from maze_transformer.training.config import ConfigHolder
from maze_transformer.training.mazedataset import MazeDatasetConfig
from maze_transformer.training.tokenizer import SPECIAL_TOKENS, MazeTokenizer
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


#! Soon to be defunct
def predict_tokens(
    cfg: ConfigHolder,
    model: HookedTransformer,
    inputs: torch.Tensor,
    n_tokens: int = 32,
    **generate_kwargs,
):
    """
    Predict the next tokens
    """
    # print(f"{inputs.shape = } {n_tokens = } {model.config.n_positions = }")
    sequence = pad_sequence(inputs, cfg)
    with torch.no_grad():
        for _ in range(n_tokens):
            sequence = model.generate(
                sequence[:, -model.config.n_positions :],
                do_sample=True,
                **generate_kwargs,
            )

    return sequence


#! Soon to be defunct
def pad_sequence(
    seq: ATensor,
    cfg: ConfigHolder,
) -> torch.Tensor:
    """pads the token according to the context length and padding token in the config"""
    # assert (padding_val is not None) or (cfg.tokenizer.pad_token is not None), \
    #        "No padding token defined in tokenizer, please provide a padding value"
    # This is super ugly
    pad_token = cfg.dataset_cfg.tokenizer_map["<PADDING>"]
    return torch.nn.functional.pad(
        seq, (cfg.dataset_cfg.seq_len_max - seq.shape[0], 0), value=pad_token
    )


def decode_maze_tokens_to_coords(
    tokens: list[str],
    mazedata_cfg: MazeDatasetConfig,
    when_noncoord: typing.Literal["except", "skip", "include"] = "skip",
) -> list[str | tuple[int, int]]:
    """given a list of tokens, decode the coordinate-tokens to a list of coordinates, leaving other tokens as-is"""
    output: list[str | tuple[int, int]] = list()
    for idx, tk in enumerate(tokens):
        if tk in mazedata_cfg.token_node_map:
            output.append(mazedata_cfg.token_node_map[tk])
        else:
            if when_noncoord == "skip":
                continue
            elif when_noncoord == "include":
                output.append(tk)
            elif when_noncoord == "except":
                raise ValueError(f"token '{tk}' at {idx = } is not a coordinate")
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
    path_start_idx: int = tokens.index(path_start_token) + 1
    maze_tokens: list[str] = tokens[:path_start_idx]
    path_true_tokens: list[str] = tokens[path_start_idx:]

    if include_start_coord:
        # add the first coordinate to `maze_tokens`
        maze_tokens.append(path_true_tokens[0])

    # encode + pad the maze tokens
    #! TODO update once tokenizer changes
    maze_arr_nopad: ATensor = torch.tensor(
        [data_cfg.tokenizer_map[t] for t in maze_tokens],
        dtype=torch.long,
    )
    eos_token_id = data_cfg.tokenizer_map[SPECIAL_TOKENS["path_end"]]

    # have the model predict some tokens
    maze_arr_nopad = maze_arr_nopad.unsqueeze(0)
    if verbose:
        print("Generating Model Completions")
    #! NOTE verbose flag here will require latest clone of TrasformerLens from github
    predictions = model.generate(
        maze_arr_nopad,
        eos_token_id=eos_token_id,
        stop_at_eos=True,
        max_new_tokens=n_tokens_pred,
        verbose=verbose,
    )

    # decode the tokens
    predicted_and_context_tokens: list[str] = [
        data_cfg.token_arr[t] for t in predictions[0]
    ]
    pac_path_start_idx: int = predicted_and_context_tokens.index(path_start_token) + 1
    predicted_tokens: list[str] = predicted_and_context_tokens[pac_path_start_idx:]

    if verbose:
        print(
            f"{maze_tokens = }\n{path_true_tokens = }\n{predicted_and_context_tokens = }\n{predicted_tokens = }"
        )

    # convert tokens to coordinates
    path_true: list[tuple[int, int]] = decode_maze_tokens_to_coords(
        path_true_tokens,
        mazedata_cfg=data_cfg,
        when_noncoord="skip",
    )

    path_predicted: list[tuple[int, int]] = decode_maze_tokens_to_coords(
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
