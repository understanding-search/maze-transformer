# Generic
import os
from pathlib import Path
from copy import deepcopy
import typing

# Numerical Computing
import numpy as np
import torch
import pandas as pd
# import torch.nn.functional as F
from fancy_einsum import einsum
import einops
from jaxtyping import Float, Int, Bool
import matplotlib.pyplot as plt

from muutils.misc import shorten_numerical_to_str
from muutils.nbutils.configure_notebook import configure_notebook
# TransformerLens imports
from transformer_lens import ActivationCache

# Our Code
# dataset stuff
from maze_dataset import MazeDataset, MazeDatasetConfig, SolvedMaze, LatticeMaze, SPECIAL_TOKENS
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode
from maze_dataset.plotting.print_tokens import color_maze_tokens_AOTP

# model stuff
from maze_transformer.training.config import ConfigHolder, ZanjHookedTransformer, BaseGPTConfig

# mechinterp stuff
from maze_transformer.mechinterp.plot_logits import plot_logits
from maze_transformer.mechinterp.logit_attrib_task import DLAProtocol, DLAProtocolFixed, token_after_fixed_start_token
from maze_transformer.mechinterp.logit_diff import logits_diff_multi, logit_diff_orig, logit_diff_residual_stream, residual_stack_to_logit_diff

def compute_logit_lens(
    model: ZanjHookedTransformer,
    cache: ActivationCache,
    answer_tokens: Int[torch.Tensor, "n_mazes"],        
) -> tuple[
    torch.Tensor, torch.Tensor, # x/y for diff
    torch.Tensor, torch.Tensor, # x/y for attr
]:
    # logit diff
    avg_diff, diff_direction = logit_diff_residual_stream(
        model=model,
        cache=cache,
        answer_tokens=answer_tokens,
        compare_to=None,
        directions=True,
    )

    accumulated_residual, labels = cache.accumulated_resid(
        layer=-1, incl_mid=True, pos_slice=-1, return_labels=True,
    )

    logit_lens_logit_diffs = residual_stack_to_logit_diff(
        residual_stack=accumulated_residual,
        cache=cache,
        logit_diff_directions=diff_direction,
    )

    # logit attribution
    per_layer_residual, labels = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
    per_layer_logit_diffs = residual_stack_to_logit_diff(
        residual_stack=per_layer_residual,
        cache=cache,
        logit_diff_directions=diff_direction,
    )

    return (
        # np.arange(model.zanj_model_config.model_cfg.n_layers*2+1)/2,
        np.arange(logit_lens_logit_diffs.shape[0]),
        logit_lens_logit_diffs.to("cpu").numpy(),
        np.arange(per_layer_logit_diffs.shape[0]),
        per_layer_logit_diffs.to("cpu").numpy(),
    )

def plot_logit_lens(
        model: ZanjHookedTransformer,
        cache: ActivationCache,
        answer_tokens: Int[torch.Tensor, "n_mazes"],
    ) -> tuple[
        tuple[plt.Figure, plt.Axes], # figure and axes
        tuple[
            torch.Tensor, torch.Tensor, # x/y for diff
            torch.Tensor, torch.Tensor, # x/y for attr
        ],
    ]:
    # set up figure
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax_diff, ax_attr = ax

    diff_x, diff_y, attr_x, attr_y = compute_logit_lens(
        model=model,
        cache=cache,
        answer_tokens=answer_tokens,
    )
    
    ax_diff.plot(diff_x, diff_y)
    ax_diff.set_title("Logit Difference from Accumulated Residual Stream")
    ax_diff.set_xlabel("Layer")
    ax_diff.set_ylabel("Logit Difference")

    ax_attr.plot(attr_x, attr_y)
    ax_attr.set_title("Logit Attribution from Residual Stream")
    ax_attr.set_xlabel("Layer")
    ax_attr.set_ylabel("Logit Attribution")

    plt.show()

    return (fig, ax), (diff_x, diff_y, attr_x, attr_y)