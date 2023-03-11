import os
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.io as pio
import torch
from IPython import get_ipython


def initialize_notebook(run_path=Path("../data/maze/g4-n10"), seed=42, dark_mode=True):
    """Shared Jupyter notebook setup steps:
    - Set random seed
    - Validate specified model path
    - Set device based on availability
    - Set module reloading before code execution
    - Set plot rendering and formatting
    - Disable PyTorch gradients
    """

    # Set seed for reproducibility
    _ = torch.manual_seed(seed)

    # Get latest model
    assert run_path.exists(), f"Run path {run_path.as_posix()} does not exist"
    model_path = list(sorted(run_path.glob("**/model.final.pt"), key=os.path.getmtime))[
        -1
    ].resolve()

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device set to MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device set to CUDA")
    else:
        device = torch.device("cpu")
        print("Device set to CPU")

    # Reload modules before executing user code
    ipython = get_ipython()
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

    # Specify plotly renderer for vscode
    pio.renderers.default = "notebook_connected"

    if dark_mode:
        pio.templates.default = "plotly_dark"
        plt.style.use("dark_background")

    # We won't be training any models
    torch.set_grad_enabled(False)

    return device, model_path
