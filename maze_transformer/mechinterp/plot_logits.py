# Numerical Computing
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float, Int

# Our Code
from maze_dataset.tokenization import MazeTokenizer

_DEFAULT_SUBPLOTS_KWARGS: dict = dict(
    figsize=(20, 20),
    height_ratios=[3, 1],
)


def plot_logits(
    last_tok_logits: Float[torch.Tensor, "n_mazes d_vocab"],
    target_idxs: Int[torch.Tensor, "n_mazes"],
    tokenizer: MazeTokenizer,
    n_bins: int = 50,
    mark_incorrect: bool = True,
    mark_correct: bool = False,
    subplots_kwargs: dict | None = None,
    show: bool = True,
) -> None:
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
        # place yellow x at max logit token
        ax_all.scatter(
            last_tok_logits.argmax(dim=1),
            np.arange(n_mazes),
            marker="x",
            color="yellow",
        )
        # place red dot at correct token
        ax_all.scatter(target_idxs, np.arange(n_mazes), marker=".", color="red")
        if mark_incorrect:
            raise ValueError("mark_correct and mark_incorrect cannot both be True")

    if mark_incorrect:
        # place a red dot wherever the max logit token is not the correct token
        ax_all.scatter(
            last_tok_logits.argmax(dim=1)[last_tok_logits.argmax(dim=1) != target_idxs],
            np.arange(n_mazes)[last_tok_logits.argmax(dim=1) != target_idxs],
            marker=".",
            color="red",
        )

    # histogram of logits for correct and incorrect tokens
    # --------------------------------------------------
    ax_sum.set_ylabel("probability density")
    ax_sum.set_xlabel("logit value")

    # get correct token logits
    correct_token_logits: Float[torch.Tensor, "n_mazes"] = torch.gather(
        last_tok_logits, 1, target_idxs.unsqueeze(1)
    ).squeeze(1)
    mask = torch.ones(n_mazes, d_vocab, dtype=torch.bool)
    mask.scatter_(1, target_idxs.unsqueeze(1), False)
    other_token_logits: Float[torch.Tensor, "n_mazes d_vocab-1"] = last_tok_logits[
        mask
    ].reshape(n_mazes, d_vocab - 1)

    # plot histogram
    bins: Float[np.ndarray, "n_bins"] = np.linspace(
        last_tok_logits.min(), last_tok_logits.max(), n_bins
    )
    ax_sum.hist(
        correct_token_logits.numpy(),
        density=True,
        bins=bins,
        label="correct token",
    )
    ax_sum.hist(
        other_token_logits.numpy().flatten(),
        density=True,
        bins=bins,
        label="other token",
    )
    ax_sum.legend()

    if show:
        plt.show()

    return fig, (ax_all, ax_sum)
