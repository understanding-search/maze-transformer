import json
from pathlib import Path
import typing
from typing import Any, NamedTuple

import numpy as np
import torch
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTConfig
from muutils.json_serialize import json_serialize
from muutils.tensor_utils import ATensor, NDArray
from muutils.misc import shorten_numerical_to_str

from maze_transformer.generation.latticemaze import LatticeMaze, CoordTup
from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.training.tokenizer import MazeTokenizer, SPECIAL_TOKENS
from maze_transformer.training.mazedataset import MazeDatasetConfig
from maze_transformer.evaluation.plot_maze import plot_multi_paths, PathFormat
from maze_transformer.training.dataset import GPTDatasetConfig
from maze_transformer.training.config import TrainConfig
from maze_transformer.training.training import TRAIN_SAVE_FILES

# pylint: disable=protected-access

MazePath = list[CoordTup]
ArrMazePath = NDArray[((Any, "node"), (2, "xypos")), int]


def check_configs_present(folder: Path) -> bool:
    return (folder / TRAIN_SAVE_FILES.data_cfg).exists() and (
        folder / TRAIN_SAVE_FILES.train_cfg
    ).exists()


LoadedModelConfigs = NamedTuple(
    "LoadedModelConfigs",
    [
        ("data_cfg", GPTDatasetConfig),
        ("train_cfg", TrainConfig),
        ("model_cfg", OpenAIGPTConfig),
        ("model", OpenAIGPTLMHeadModel),
    ],
)


def load_model_with_configs(
    model_path: str, data_cfg_class: type, verbose: bool = False
) -> LoadedModelConfigs:
    """
    Load a model and associated config files from a path.
    """

    # TODO: make this less fragile
    # load the configs
    # get path to the folder containing the model
    config_folder: Path = Path(model_path).parent
    # check for the filenames, go up a dir if they don't exist
    if not check_configs_present(config_folder):
        config_folder = config_folder.parent
        assert check_configs_present(
            config_folder
        ), f"Couldn't find configs in directory of or parent directory of {model_path}"

    # load the configs
    with open(config_folder / TRAIN_SAVE_FILES.train_cfg, "r") as f:
        train_cfg_raw: dict = json.load(f)

    train_cfg: TrainConfig = TrainConfig.load(train_cfg_raw)
    if verbose:
        print(f"{train_cfg = }")
        print("-" * 40)

    model_cfg: OpenAIGPTConfig = train_cfg.get_gpt_config()
    if verbose:
        print("model_cfg = ", json_serialize(model_cfg.to_dict(), error_mode="warn"))
        print("-" * 40)

    with open(config_folder / TRAIN_SAVE_FILES.data_cfg, "r") as f:
        data_cfg: GPTDatasetConfig = data_cfg_class.load(json.load(f))

    model: OpenAIGPTLMHeadModel = OpenAIGPTLMHeadModel(model_cfg)
    state_dict: dict = torch.load(model_path)
    # print(state_dict.keys())
    model.load_state_dict(state_dict)
    model.eval()

    print(
        f"loaded model with {shorten_numerical_to_str(model.num_parameters())} parameters"
    )

    return LoadedModelConfigs(
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        model=model,
    )


def predict_tokens(
    model: OpenAIGPTLMHeadModel,
    inputs: ATensor,
    n_tokens: int = 32,
    **generate_kwargs,
):
    """
    Predict the next tokens
    """
    # print(f"{inputs.shape = } {n_tokens = } {model.config.n_positions = }")
    sequence = pad_sequence(inputs, model.config)
    with torch.no_grad():
        for _ in range(n_tokens):
            sequence = model.generate(
                sequence[:, -model.config.n_positions :],
                do_sample=True,
                **generate_kwargs,
            )

    return sequence


def pad_sequence(seq: ATensor, model_cfg: OpenAIGPTConfig) -> ATensor:
    """pads the token according to the context length and padding token in the config"""
    return torch.nn.functional.pad(
        seq,
        (model_cfg.n_positions - seq.shape[0], 0),
        value=model_cfg.pad_token_id,
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
    model: OpenAIGPTLMHeadModel,
    n_tokens_pred: int,
    include_start_coord: bool = True,
    verbose: bool = False,
    **generate_kwargs,
) -> tuple[LatticeMaze, MazePath, MazePath]:
    """given tokens from a dataset, predict the next tokens with the model, decode both true and predicted to paths

    ### Parameters:
     - `tokens : list[str]`
       raw tokens from dataset, containing both maze and true path
     - `data_cfg : MazeDatasetConfig`
       config for the dataset
     - `model : OpenAIGPTLMHeadModel`
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
    path_start_token: str = SPECIAL_TOKENS["start_path"]
    path_start_idx: int = tokens.index(path_start_token) + 1
    maze_tokens: list[str] = tokens[:path_start_idx]
    path_true_tokens: list[str] = tokens[path_start_idx:]

    if include_start_coord:
        # add the first coordinate to `maze_tokens`
        maze_tokens.append(path_true_tokens[0])

    # encode + pad the maze tokens
    maze_arr_nopad: ATensor = torch.tensor(
        [data_cfg.tokenizer_map[t] for t in maze_tokens],
        dtype=torch.int32,
        device="cpu",
    )
    # maze_arr: ATensor = pad_sequence(maze_arr_nopad, model.config)

    # have the model predict some tokens
    predictions: ATensor = predict_tokens(
        model=model,
        inputs=maze_arr_nopad.unsqueeze(0),
        n_tokens=n_tokens_pred,
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


def generate_plot_predicted_path(
    model_path: str,
    n_tokens_pred: int = 5,
):
    data_cfg: MazeDatasetConfig
    train_cfg: TrainConfig
    model_cfg: OpenAIGPTConfig
    model: OpenAIGPTLMHeadModel
    loaded_model_and_configs: LoadedModelConfigs = load_model_with_configs(
        model_path, MazeDatasetConfig
    )
    data_cfg, train_cfg, model_cfg, model = loaded_model_and_configs

    # generate a maze
    grid_n: int = data_cfg.grid_n
    maze: LatticeMaze = LatticeMazeGenerators.gen_dfs((grid_n, grid_n))
    c_start = (0, 0)
    c_end = (grid_n - 1, grid_n - 1)

    # solve the maze explicitly
    path_true = np.array(
        maze.find_shortest_path(
            c_start=c_start,
            c_end=c_end,
        )
    )

    solved_maze: MazeTokenizer = MazeTokenizer(
        maze=maze,
        solution=np.array(
            maze.find_shortest_path(
                c_start=c_start,
                c_end=c_end,
            )
        ),
    )

    # tokenize the maze
    maze_only_tokens: list[str] = solved_maze.as_tokens(
        data_cfg.node_token_map, solution=False
    ) + [SPECIAL_TOKENS["start_path"]]

    print("maze tokens:", maze_only_tokens)

    array_nopad = torch.tensor(
        [data_cfg.tokenizer_map[t] for t in maze_only_tokens],
        dtype=torch.int32,
        device="cpu",
    )

    array: ATensor = pad_sequence(array_nopad, model_cfg)

    # have the model predict some tokens
    predictions = predict_tokens(model, array.unsqueeze(0), n_tokens_pred)

    print(predictions)

    # decode the tokens
    predicted_tokens = [data_cfg.token_arr[t] for t in predictions[0]]

    print(predicted_tokens)

    path_predicted: list[tuple[int, int]] = decode_maze_tokens_to_coords(
        predicted_tokens[len(maze_only_tokens) :],
        mazedata_cfg=data_cfg,
        when_noncoord="skip",
    )

    # plot the maze and both solutions
    # for label, fmt, color, path in paths
    plot_multi_paths(
        maze=maze,
        paths=[
            PathFormat(path_true, "true", "-", "red", {"width": 0.015}),
            PathFormat(np.array(path_predicted), "predicted", ":", "blue", {}),
        ],
    )
