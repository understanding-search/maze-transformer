# Numerical Computing
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from maze_dataset import CoordTup

# Our Code
from maze_dataset.tokenization import MazeTokenizer

_DEFAULT_SUBPLOTS_KWARGS: dict = dict(
    figsize=(20, 20),
    height_ratios=[3, 1],
)


def plot_logit_histograms(
    last_tok_logits: Float[torch.Tensor, "n_mazes d_vocab"],
    target_idxs: Int[torch.Tensor, "n_mazes"],
    token_groups: Bool[torch.Tensor, "n_mazes d_vocab"] | None = None,
    show_all_other_tokens: bool = True,
    n_bins: int = 50,
    ax: plt.Axes | None = None,
    density: bool = True,
    logy: bool = False,
) -> plt.Axes:
    n_mazes: int
    d_vocab: int
    n_mazes, d_vocab = last_tok_logits.shape

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.set_ylabel("probability density" if density else "frequency")
    ax.set_xlabel("logit value")

    # get correct token logits
    correct_token_logits: Float[torch.Tensor, "n_mazes"] = torch.gather(
        last_tok_logits, 1, target_idxs.unsqueeze(1)
    ).squeeze(1)
    mask = torch.ones(n_mazes, d_vocab, dtype=torch.bool)
    mask.scatter_(1, target_idxs.unsqueeze(1), False)

    # specified token groups
    token_groups_logits: dict[str, Float[torch.Tensor, "n_mazes sub_vocab"]] = dict()
    if token_groups is not None:
        for group_name, group_mask in token_groups.items():
            if group_mask.sum() > 0:
                token_groups_logits[group_name] = last_tok_logits[group_mask].flatten()
                # remove tokens in this group from mask
                mask = mask & ~group_mask

    # all other tokens
    if show_all_other_tokens:
        other_token_logits: Float[torch.Tensor, "n_mazes d_vocab-1"] = last_tok_logits[
            mask
        ].flatten()
        token_groups_logits["all other tokens"] = other_token_logits

    # plot histogram
    bins: Float[np.ndarray, "n_bins"] = np.linspace(
        last_tok_logits.min(), last_tok_logits.max(), n_bins
    )
    ax.hist(
        correct_token_logits.numpy(),
        density=density,
        bins=bins,
        label="correct tokens",
        alpha=0.5,
        hatch="/",
    )
    for group_name, group_logits in token_groups_logits.items():
        ax.hist(
            group_logits.numpy(),
            density=density,
            bins=bins,
            label=group_name,
            alpha=0.3,
        )
    ax.legend()
    if logy:
        ax.set_yscale("log")

    return ax


def get_baseline_incorrect_group(
    prompts: list[list[str]],
    tokenizer: MazeTokenizer,
    baseline: "RandomBaseline",
) -> Bool[torch.Tensor, "n_mazes d_vocab"]:
    """
    Returns a mask of shape (n_mazes, d_vocab) where each row is True for the incorrect but valid next tokens
    """
    n_mazes: int = len(prompts)
    d_vocab: int = tokenizer._vocab_size

    correct_steps: list[CoordTup | str] = list()
    incorrect_steps: list[CoordTup | str] = list()

    for p in prompts:
        corr_s, incor_s = baseline.get_valid_next_steps(p)
        correct_steps.append(corr_s)
        incorrect_steps.append(incor_s)

    incorrect_mask: Bool[torch.Tensor, "n_mazes d_vocab"] = torch.zeros(
        n_mazes, d_vocab, dtype=torch.bool
    )

    for i, (corr_s, incor_s) in enumerate(zip(correct_steps, incorrect_steps)):
        incorrect_mask[i, tokenizer.encode(tokenizer.coords_to_strings(incor_s))] = True

    return incorrect_mask


def plot_logits(
    last_tok_logits: Float[torch.Tensor, "n_mazes d_vocab"],
    target_idxs: Int[torch.Tensor, "n_mazes"],
    tokenizer: MazeTokenizer,
    n_bins: int = 50,
    mark_incorrect: bool = False,
    mark_correct: bool = True,
    subplots_kwargs: dict | None = None,
    show: bool = True,
    density: bool = True,
    logy: bool = False,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    # set up figure
    # --------------------------------------------------
    n_mazes: int
    d_vocab: int
    n_mazes, d_vocab = last_tok_logits.shape
    if subplots_kwargs is None:
        subplots_kwargs = _DEFAULT_SUBPLOTS_KWARGS

    fig, (ax_all, ax_sum) = plt.subplots(
        2, 1, **{**_DEFAULT_SUBPLOTS_KWARGS, **subplots_kwargs}
    )

    # fig.subplots_adjust(hspace=0.5, bottom=0.1, top=0.9, left=0.1, right=0.9)

    # plot heatmap of logits
    # --------------------------------------------------
    # all vocab elements
    ax_all.set_xlabel("vocab element logit")
    ax_all.set_ylabel("maze index")
    # add vocab as xticks
    ax_all.set_xticks(ticks=np.arange(d_vocab), labels=tokenizer.token_arr, rotation=90)
    ax_all.imshow(last_tok_logits.numpy(), aspect="auto")
    # set colorbar
    plt.colorbar(ax_all.imshow(last_tok_logits.numpy(), aspect="auto"), ax=ax_all)

    if mark_correct:
        # place red dot at max logit token
        ax_all.scatter(
            last_tok_logits.argmax(dim=1),
            np.arange(n_mazes),
            marker=".",
            color="red",
        )
        # place red + at correct token
        ax_all.scatter(target_idxs, np.arange(n_mazes), marker="+", color="red")
        if mark_incorrect:
            raise ValueError("mark_correct and mark_incorrect cannot both be True")

    if mark_incorrect:
        # place a red x wherever the max logit token is not the correct token
        ax_all.scatter(
            last_tok_logits.argmax(dim=1)[last_tok_logits.argmax(dim=1) != target_idxs],
            np.arange(n_mazes)[last_tok_logits.argmax(dim=1) != target_idxs],
            marker="x",
            color="red",
        )
        # place red x at correct token
        ax_all.scatter(target_idxs, np.arange(n_mazes), marker="x", color="red")

    # histogram of logits for correct and incorrect tokens
    # --------------------------------------------------
    plot_logit_histograms(
        last_tok_logits=last_tok_logits,
        target_idxs=target_idxs,
        ax=ax_sum,
        n_bins=n_bins,
        density=density,
        logy=logy,
    )

    if show:
        plt.show()

    return fig, (ax_all, ax_sum)


def plot_logits_maze(*args, **kwargs):
    raise NotImplementedError()
