import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
from jaxtyping import Float, Int

# maze dataset
from maze_dataset import (
    SPECIAL_TOKENS,
    CoordTup,
    MazeDataset,
    MazeDatasetConfig,
    SolvedMaze,
)
from maze_dataset.tokenization import MazeTokenizer
from maze_dataset.tokenization.token_utils import (
    get_context_tokens,
    get_path_tokens,
    remove_padding_from_token_str,
)
from maze_dataset.tokenization.util import strings_to_coords
from maze_dataset.utils import WhenMissing

# muutils
from muutils.mlutils import chunks
from muutils.statcounter import StatCounter

# TransformerLens
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils

from maze_transformer.evaluation.path_evals import PathEvalFunction, PathEvals
from maze_transformer.tokenizer import HuggingMazeTokenizer
from maze_transformer.training.config import ConfigHolder
from maze_transformer.training.train_save_files import TRAIN_SAVE_FILES
from maze_transformer.utils.padding import pad_and_batch_tensors

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
    fold_ln: bool = True,
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

    # TODO Make this part of the ConfigHolder - https://github.com/understanding-search/maze-transformer/issues/31

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

    model.process_weights_(fold_ln=fold_ln)
    model.setup()  # Re-attach layernorm hooks by calling setup
    model.eval()

    return model, config_holder


def predict_maze_paths(
    tokens_batch: list[list[str]],
    data_cfg: MazeDatasetConfig,
    model: HookedTransformer,
    # Note: The start coord is included in the model input, so max_new_tokens is how many tokens can be predicted AFTER the start. This function returns the full paths, including start coord, so in general the max returned path length is max_new_tokens + 1
    max_new_tokens: int | None = 8,
    smart_max_new_tokens: bool = False,
    verbose: bool = False,
    when_noncoord: WhenMissing = "skip",
    temperature: float = 0.0,
    batch_size: int | None = None,
) -> list[list[str | tuple[int, int]]]:
    """given the model and a batch of context tokens, make predictions for the path"""

    # check types
    assert isinstance(
        tokens_batch, (list, tuple)
    ), f"tokens_batch must be a list, got {type(tokens_batch)}"
    assert all(
        isinstance(tokens, (list, tuple)) for tokens in tokens_batch
    ), f"tokens_batch must be a list of lists, got {[type(tokens) for tokens in tokens_batch] = }"
    assert all(
        isinstance(x, str) for tokens in tokens_batch for x in tokens
    ), f"tokens_batch must be a list of lists of strings, got {[type(x) for tokens in tokens_batch for x in tokens] = }"

    if max_new_tokens is None:
        assert (
            smart_max_new_tokens
        ), "if max_new_tokens is None, smart_max_new_tokens must be True"

    maze_tokenizer: MazeTokenizer = model.tokenizer._maze_tokenizer

    contexts_lists: list[list[str]] = [
        get_context_tokens(tokens) for tokens in tokens_batch
    ]
    contexts_strings: list[str] = [" ".join(tokens) for tokens in contexts_lists]
    contexts_tokens: list[list[int]] = [
        maze_tokenizer.encode(x) for x in contexts_lists
    ]

    predictions_out: list[list[str]] = list()

    generate_kwargs: dict = dict(
        eos_token_id=model.tokenizer._tokenizer_map[SPECIAL_TOKENS.PATH_END],
        stop_at_eos=True,
        verbose=verbose,
        # return_type="str",
    )
    if temperature != 0.0:
        generate_kwargs["temperature"] = temperature
    else:
        generate_kwargs["top_k"] = 1

    if batch_size is not None:
        # tensor, pad, and batch the tokens
        contexts_tensored: list[Int[torch.Tensor, "batch pos"]] = pad_and_batch_tensors(
            contexts_tokens=contexts_tokens,
            batch_size=batch_size,
            padding_idx=maze_tokenizer.padding_token_index,
            padding_dir="left",  # TODO: read this from model, but it breaks for the RandomBaseline
        )

        # print(f"{len(contexts_tensored) = }")
        # print(f"{[x.shape for x in contexts_tensored]}")
        # print(f"{contexts_tensored = }")

        for batch in contexts_tensored:
            if smart_max_new_tokens:
                max_new_tokens = model.cfg.n_ctx - batch.shape[1] - 1

            predictions: torch.Tensor | list[str] | list[list[str]] = model.generate(
                batch,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )

            if isinstance(predictions, torch.Tensor):
                predictions_out.extend([maze_tokenizer.decode(x) for x in predictions])
            elif isinstance(predictions, list):
                # assume same type throughout
                if isinstance(predictions[0], str):
                    predictions_out.extend([x.split(" ") for x in predictions])
                elif isinstance(predictions[0], list):
                    predictions_out.extend(predictions)
            else:
                raise TypeError(
                    f"Unexpected type for predictions: {type(predictions)}\n{predictions = }"
                )

    else:
        # pass string prompts one at a time
        for i, context in enumerate(contexts_strings):
            if smart_max_new_tokens:
                max_new_tokens = model.cfg.n_ctx - len(contexts_lists[i]) - 1

            prediction: str = model.generate(
                context,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )

            predictions_out.append(prediction.split(" "))

    # turn the predicted tokens into paths
    paths: list[list[str | tuple[int, int]]] = []
    for pred_tokens in predictions_out:
        # get the sequence from the path_start to the first path_end
        # (since the latter is also our EOS token)
        path_start_idx: int = pred_tokens.index(SPECIAL_TOKENS.PATH_START)
        path_end_idx: int
        try:
            path_end_idx = (
                pred_tokens.index(SPECIAL_TOKENS.PATH_END, path_start_idx) + 1
            )
        except ValueError:
            path_end_idx = len(pred_tokens)
        path_tokens: list[str] = pred_tokens[path_start_idx:path_end_idx]
        path_coords: list[str | CoordTup] = strings_to_coords(
            path_tokens,
            when_noncoord=when_noncoord,
        )
        paths.append(path_coords)

    return paths


def evaluate_path_predictions(
    solved_mazes: list[SolvedMaze],
    predictions: list[list[tuple[int, int]]],
    path_evals: dict[str, PathEvalFunction],
) -> dict[str, StatCounter]:
    path_scores: dict[str, StatCounter] = {
        name: StatCounter() for name in path_evals.keys()
    }
    for name, func in path_evals.items():
        path_scores[name].update(
            func(
                maze=solved_maze.maze,
                solution=np.array(solved_maze.solution),
                prediction=np.array(prediction),
            )
            for solved_maze, prediction in zip(solved_mazes, predictions)
        )

    return path_scores


def evaluate_model(
    model: HookedTransformer,
    dataset: MazeDataset,
    dataset_tokens: list[list[str]] | None = None,
    eval_functions: dict[str, PathEvalFunction] | None = None,
    max_new_tokens: int = 8,
    batch_size: int = 64,
    verbose: bool = False,
) -> dict[str, StatCounter]:
    """Run a set of eval functions on a model for a given dataset. Returns a seperate StatCounter for each eval function.

    if dataset_tokens is provided, we assume that the dataset has already been tokenized and we skip tokenization. MAKE SURE THERE IS NOT A MISMATCH BETWEEN THE DATASET AND DATASET_TOKENS
    """

    if not eval_functions:
        # TODO: potentially model evals which aren't path evals?
        eval_functions = PathEvals.EVALS

    score_counters: dict[str, StatCounter] = {
        name: StatCounter() for name in eval_functions
    }

    if dataset_tokens is None:
        dataset_tokens = dataset.as_tokens(
            model.tokenizer._maze_tokenizer, join_tokens_individual_maze=False
        )
    else:
        assert len(dataset) == len(
            dataset_tokens
        ), f"dataset and dataset_tokens must be the same length and must be from corresponding mazes, got {len(dataset) = } and {len(dataset_tokens) = }"

    for batch in chunks(zip(dataset, dataset_tokens), batch_size):
        maze_batch, tokens_batch = zip(*batch)
        predictions: list[str | list[tuple[int, int]]] = predict_maze_paths(
            tokens_batch=tokens_batch,
            data_cfg=dataset.cfg,
            model=model,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
        )

        for name, func in eval_functions.items():
            score_counters[name].update(
                func(
                    maze=solved_maze,
                    solution=np.array(solved_maze.solution),
                    prediction=np.array(prediction),
                    model=model,
                )
                for solved_maze, prediction in zip(maze_batch, predictions)
            )

    return score_counters


def evaluate_logits(
    logits: Float[torch.Tensor, "batch pos d_vocab"],
    batch: list[int],
    config: ConfigHolder,
    tokenizer: HuggingMazeTokenizer,
    path_evals: dict[str, PathEvalFunction] | None = None,
) -> dict[str, StatCounter]:
    """Runs a set of eval functions on the provided logits. For path evals, an attempt will be made to extract a predicted path from the logits (it is assumed that the logits are an entire sequence output from training, so they contain the adj_list plus path)"""

    raise NotImplementedError(
        "evaluate_logits does not function correctly, and at the moment there are only path evals anyway"
    )

    scores: dict[str, StatCounter] = {}

    if path_evals:
        # TODO: this is pretty much wrong -- sampling from the logits over the sequence should not produce a valid path
        sampled_logits = tl_utils.sample_logits(logits)
        prediction_tokens = tokenizer.batch_decode(sampled_logits)
        predicted_paths = []
        for tokens in prediction_tokens:
            # this returns first path_start to end of list. Early in training there may be multiple path_start tokens, so results should be treated with caution
            path_tokens = get_path_tokens(tokens.split(" "))
            path_coords = strings_to_coords(path_tokens, when_noncoord="skip")
            predicted_paths.append(cast(list[tuple[int, int]], path_coords))

        maze_tokens = [
            remove_padding_from_token_str(token_str)
            for token_str in tokenizer.batch_decode(batch)
        ]

        solved_mazes = [
            SolvedMaze.from_tokens(tokens.split(" "), config.dataset_cfg)
            for tokens in maze_tokens
        ]
        scores |= evaluate_path_predictions(solved_mazes, predicted_paths, path_evals)

    return scores
