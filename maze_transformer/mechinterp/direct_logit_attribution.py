import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float, Int

# TransformerLens imports
from transformer_lens import ActivationCache

# mechinterp stuff
from maze_transformer.mechinterp.logit_diff import (
    logit_diff_residual_stream,
    residual_stack_to_logit_diff,
)

# model stuff
from maze_transformer.training.config import ZanjHookedTransformer


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

    per_head_residual, labels = cache.stack_head_results(
        layer=-1, pos_slice=-1, return_labels=True
    )
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
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, Float[np.ndarray, "layer head"]]:
    data = compute_direct_logit_attribution(
        model=model,
        cache=cache,
        answer_tokens=answer_tokens,
    )

    data_extreme: float = np.max(np.abs(data))
    # colormap centeres on zero
    fig, ax = plt.subplots()
    ax.imshow(data, cmap="RdBu", vmin=-data_extreme, vmax=data_extreme)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    plt.colorbar(ax.get_images()[0], ax=ax)
    ax.set_title(f"Logit Difference from each head\n{model.zanj_model_config.name}")

    if show:
        plt.show()

    return fig, ax, data
