import os
import random

import numpy as np
import torch

DEFAULT_SEED = 42


def get_device():
    """Get the torch.device instance on which torch.Tensors should be allocated."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_reproducibility(seed=DEFAULT_SEED):
    """
    Improve model reproducibility. See https://github.com/NVIDIA/framework-determinism for more information.

    Deterministic operations tend to have worse performance than nondeterministic operations, so this method trades
    off performance for reproducibility. Set use_deterministic_algorithms to True to improve performance.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)
    # Ensure reproducibility for concurrent CUDA streams
    # see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    # https://stackoverflow.com/a/312464
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
