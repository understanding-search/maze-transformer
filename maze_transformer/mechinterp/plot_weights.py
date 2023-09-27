import torch
import numpy as np
from jaxtyping import Float, Int
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer

def plot_important_neurons(
        model: HookedTransformer,
        layer: int,
        neuron_idxs: Int[np.ndarray, "neuron_idxs"]|None,
        dla_data: dict[str, np.ndarray]|None,
        n_important_neurons: int = 10,
    ) -> tuple[plt.Figure, plt.Axes]:
    """Plot the weights and biases for the selected or most important neurons in a given layer
    
    - if both of `neuron_idxs` and `dla_data` are `None`, then all neurons will be plotted
    - if a value is provided for `neuron_idxs`, then only those neurons will be plotted
    - if a value is provided for `dla_data`, then the most important neurons will be selected based on the DLA data
    """
    
    # get layers from the form of the state dict
    state_dict_keys: list[str] = list(model.state_dict().keys())
    n_layers
    if neuron_idxs is None and dla_data is None:
        neuron_idxs = 

    
    # Identify important neurons based on DLA data
    important_neuron_idxs: np.ndarray = np.argsort(np.abs(dla_data["neurons"][layer]))[-n_important_neurons:][::-1]
    
    # Get the number of layers and construct the key for accessing model state
    n_layers: int = model.zanj_model_config.model_cfg.n_layers
    key: str = f"blocks.{layer}.mlp"
    
    # Cache model state for easier access
    model_state = model.state_dict()
    
    # Create named subplots
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    w_in_ax, b_in_ax, w_out_ax, dla_ax = axes
    
    # Helper function to reduce repetition in plotting
    def plot_data(ax, data, title: str, ylabel: str = None):
        im = ax.imshow(data, aspect='auto', interpolation='none', cmap='RdBu')
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        if ylabel:
            ax.set_ylabel(ylabel)
    
    # Plot in weight
    w_in_data = model_state[key + ".W_in"].cpu().numpy()[:, important_neuron_idxs]
    plot_data(w_in_ax, w_in_data, "W_in", "input neuron")
    
    # Plot in bias
    b_in_data = model_state[key + ".b_in"].cpu().numpy()[important_neuron_idxs][None, :]
    plot_data(b_in_ax, b_in_data, "b_in")
    
    # Plot out weight
    w_out_data = model_state[key + ".W_out"].cpu().numpy()[important_neuron_idxs, :].T
    plot_data(w_out_ax, w_out_data, "W_out", "output neuron")
    
    # Plot DLA
    dla_data = dla_data["neurons"][layer][important_neuron_idxs][None, :]
    plot_data(dla_ax, dla_data, "DLA")
