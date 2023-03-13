import matplotlib.pyplot as plt
import plotly.io as pio
from IPython import get_ipython

from maze_transformer.utils.utils import get_device, set_reproducibility


def configure_notebook(seed=42, dark_mode=True):
    """Shared Jupyter notebook setup steps:
    - Set random seeds and library reproducibility settings
    - Set device based on availability
    - Set module reloading before code execution
    - Set plot rendering and formatting
    """

    # Set seeds and other reproducibility-related library options
    set_reproducibility(seed)

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

    return get_device()
