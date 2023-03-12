import matplotlib.pyplot as plt
import plotly.io as pio
import torch
from IPython import get_ipython


def configure_notebook(seed=42, dark_mode=True):
    """Shared Jupyter notebook setup steps:
    - Set random seed
    - Set device based on availability
    - Set module reloading before code execution
    - Set plot rendering and formatting
    """

    # Set seed for reproducibility
    _ = torch.manual_seed(seed)

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device set to CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device set to MPS")
    else:
        device = torch.device("cpu")
        print("Device set to CPU")

    # Reload modules before executing user code
    ipython = get_ipython()
    if "IPython.extensions.autoreload" not in ipython.extension_manager.loaded:
        ipython.magic("load_ext autoreload")
        ipython.magic("autoreload 2")

    # Specify plotly renderer for vscode
    pio.renderers.default = "notebook_connected"

    if dark_mode:
        pio.templates.default = "plotly_dark"
        plt.style.use("dark_background")

    return device
