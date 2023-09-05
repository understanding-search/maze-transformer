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

def compute_direct_logit_attribution(
    model: ZanjHookedTransformer,
    cache: ActivationCache,
    answer_tokens: Int[torch.Tensor, "n_mazes"],        
):
    # logit diff
    avg_diff, diff_direction = logit_diff_residual_stream(
        model=model,
        cache=cache,
        answer_tokens=answer_tokens,
        compare_to=None,
        directions=True,
    )

    per_head_residual, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
    per_head_logit_diffs = residual_stack_to_logit_diff(
        residual_stack=per_head_residual,
        cache=cache,
        logit_diff_directions=diff_direction,
    )
    per_head_logit_diffs = einops.rearrange(
        per_head_logit_diffs, 
        "(layer head_index) -> layer head_index", 
        layer=model.zanj_model_config.model_cfg.n_layers,
        head_index=model.zanj_model_config.model_cfg.n_heads,
    )

    return per_head_logit_diffs.to("cpu").numpy()
    

def plot_direct_logit_attribution(
    model: ZanjHookedTransformer,
    cache: ActivationCache,
    answer_tokens: Int[torch.Tensor, "n_mazes"],   
):
    data = compute_direct_logit_attribution(
        model=model,
        cache=cache,
        answer_tokens=answer_tokens,
    )
        
    data_extreme: float = np.max(np.abs(data))
    # colormap centeres on zero
    plt.imshow(data, cmap = "RdBu", vmin=-data_extreme, vmax=data_extreme)
    plt.colorbar()
    plt.title("Logit Difference from each head")
    plt.show()