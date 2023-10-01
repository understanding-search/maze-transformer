import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer


def _weights_plot_helper(
    fig: plt.Figure,
    ax: plt.Axes,
    data: Float[np.ndarray, "inputs_outputs_or_1 n_interesting_neurons"],
    title: str,
    ylabel: str = None,
    cmap: str = "RdBu",
):
    # plot heatmap
    n_rows: int = data.shape[0]
    singlerow: bool = n_rows == 1
    vbound: float = np.max(np.abs(data))
    im: plt.AxesImage = ax.imshow(
        data,
        aspect="auto" if not singlerow else "equal",
        interpolation="none",
        cmap=cmap,
        vmin=-vbound,
        vmax=vbound,
    )
    # colorbar
    fig.colorbar(im, ax=ax)
    # other figure adjustments
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)

    if singlerow:
        ax.set_yticks([])


def plot_important_neurons(
    model: HookedTransformer,
    layer: int,
    neuron_idxs: Int[np.ndarray, "neuron_idxs"] | None = None,
    neuron_dla_data: Float[np.ndarray, "n_layers n_neurons"] | None = None,
    n_important_neurons: int = 10,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the weights and biases for the selected or most important neurons in a given layer

    - if both of `neuron_idxs` and `neuron_dla_data` are `None`, then all neurons will be plotted
    - if a value is provided for `neuron_idxs`, then only those neurons will be plotted
    - if a value is provided for `neuron_dla_data`, then the most important neurons will be selected based on the DLA data
    """

    # get dimension info from model state dict (expecting TransformerLens style)

    # state dict
    state_dict: dict[str, torch.Tensor] = model.state_dict()
    state_dict_keys: list[str] = list(state_dict.keys())

    # layers
    layer_ids: list[int] = sorted(
        list(
            set(
                [
                    int(key.split(".")[1])
                    for key in state_dict_keys
                    if key.startswith("blocks.")
                ]
            )
        )
    )
    n_layers: int = len(layer_ids)
    assert n_layers == max(layer_ids) + 1, f"Layers are not contiguous? {layer_ids}"
    assert layer_ids == list(range(n_layers)), f"Layers are not contiguous? {layer_ids}"
    # handle layer negative indexing
    if layer < 0:
        layer = layer_ids[layer]
    assert layer in layer_ids, f"Layer {layer} not found in {layer_ids}"

    # model dim and hidden dim
    d_model: int
    n_neurons: int
    d_model, n_neurons = state_dict[f"blocks.{layer}.mlp.W_in"].shape

    # dim checks for sanity
    assert state_dict[f"blocks.{layer}.mlp.b_in"].shape[0] == n_neurons
    assert state_dict[f"blocks.{layer}.mlp.W_out"].shape[0] == n_neurons
    assert state_dict[f"blocks.{layer}.mlp.W_out"].shape[1] == d_model
    assert state_dict[f"blocks.{layer}.mlp.b_out"].shape[0] == d_model

    # get the neuron indices to plot

    # all neurons if nothing specified
    if neuron_idxs is None and neuron_dla_data is None:
        neuron_idxs = np.arange(n_neurons)

    # from dla data
    if neuron_dla_data is not None:
        assert (
            neuron_idxs is None
        ), "Cannot provide both neuron_idxs and neuron_dla_data"

        neuron_idxs: np.ndarray = np.argsort(np.abs(neuron_dla_data[layer]))[
            -n_important_neurons:
        ][::-1]

    mlp_key_base: str = f"blocks.{layer}.mlp"

    # Cache model state for easier access
    model_state = model.state_dict()

    # Create named subplots, tight layout
    fig, axes = plt.subplots(
        3 + int(neuron_dla_data is not None),  # w_in, b_in, w_out, dla (if applicable)
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw=dict(hspace=0.1, wspace=0.1),
    )

    if neuron_dla_data is not None:
        ax_w_in, ax_b_in, ax_w_out, ax_dla = axes
    else:
        ax_w_in, ax_b_in, ax_w_out = axes

    # Plot in weight
    w_in_data = model_state[mlp_key_base + ".W_in"].cpu().numpy()[:, neuron_idxs]
    _weights_plot_helper(fig, ax_w_in, w_in_data, "W_in", "input neuron")

    # Plot in bias
    b_in_data = model_state[mlp_key_base + ".b_in"].cpu().numpy()[neuron_idxs][None, :]
    _weights_plot_helper(fig, ax_b_in, b_in_data, "b_in")

    # Plot out weight
    w_out_data = model_state[mlp_key_base + ".W_out"].cpu().numpy()[neuron_idxs, :].T
    _weights_plot_helper(fig, ax_w_out, w_out_data, "W_out", "output neuron")

    # Plot DLA
    neuron_dla_data = neuron_dla_data[layer][neuron_idxs][None, :]
    _weights_plot_helper(fig, ax_dla, neuron_dla_data, "DLA")

    # Show the plot
    if show:
        plt.show()

    return fig, axes


def plot_embeddings(
    model: HookedTransformer, token_arr: list[str], show: bool = True
) -> tuple[plt.Figure, plt.Axes]:
    # Get the weight matrices for vocab and positional embeddings
    W_E: Float[torch.Tensor, "vocab_size d_model"] = model.W_E
    W_pos: Float[torch.Tensor, "max_seq_len d_model"] = model.W_pos

    # Make sure they have the same dimension
    d_model: int = W_E.shape[1]
    assert W_pos.shape[1] == d_model

    # Create the figure and axes
    fig, (ax_e, ax_pos) = plt.subplots(2, 1, figsize=(16, 16), sharex=True)

    # Visualize vocab embeddings
    vbound_e: float = W_E.abs().max().item()
    ax_e.imshow(
        W_E.cpu().detach().numpy(),
        cmap="RdBu",
        aspect="auto",
        vmin=-vbound_e,
        vmax=vbound_e,
    )
    ax_e.set_title("Vocab Embeddings")
    ax_e.set_ylabel("vocab item")
    ax_e.set_yticks(np.arange(len(token_arr)))
    ax_e.set_yticklabels(token_arr, fontsize=5)
    fig.colorbar(ax_e.get_images()[0], ax=ax_e)

    # Visualize positional embeddings
    vbound_pos: float = W_pos.abs().max().item()
    ax_pos.imshow(
        W_pos.cpu().detach().numpy(),
        cmap="RdBu",
        aspect="auto",
        vmin=-vbound_pos,
        vmax=vbound_pos,
    )
    ax_pos.set_title("Positional Embeddings")
    ax_pos.set_ylabel("pos vs token embed")
    ax_pos.set_xlabel("d_model")
    fig.colorbar(ax_pos.get_images()[0], ax=ax_pos)

    # Show the plot
    if show:
        plt.show()

    return fig, (ax_e, ax_pos)
