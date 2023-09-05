import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Int

# TransformerLens imports
from transformer_lens import ActivationCache

# mechinterp stuff
from maze_transformer.mechinterp.logit_diff import (
    logit_diff_residual_stream,
    residual_stack_to_logit_diff,
)

# model stuff
from maze_transformer.training.config import ZanjHookedTransformer


def compute_logit_lens(
    model: ZanjHookedTransformer,
    cache: ActivationCache,
    answer_tokens: Int[torch.Tensor, "n_mazes"],
) -> tuple[
    torch.Tensor,
    torch.Tensor,  # x/y for diff
    torch.Tensor,
    torch.Tensor,  # x/y for attr
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
        layer=-1,
        incl_mid=True,
        pos_slice=-1,
        return_labels=True,
    )

    logit_lens_logit_diffs = residual_stack_to_logit_diff(
        residual_stack=accumulated_residual,
        cache=cache,
        logit_diff_directions=diff_direction,
    )

    # logit attribution
    per_layer_residual, labels = cache.decompose_resid(
        layer=-1, pos_slice=-1, return_labels=True
    )
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
    tuple[plt.Figure, plt.Axes],  # figure and axes
    tuple[
        torch.Tensor,
        torch.Tensor,  # x/y for diff
        torch.Tensor,
        torch.Tensor,  # x/y for attr
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
