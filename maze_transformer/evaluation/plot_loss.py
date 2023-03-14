from typing import Sequence, Tuple

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
from muutils.logger.log_util import (  # type: ignore[import]
    gather_stream,
    gather_val,
    get_any_from_stream,
)


def parse_argument_list(
    arg: int | Sequence[int] | str,
    delimiter: str | None = ",",
    typecast: type = int,
) -> list[int]:
    if isinstance(arg, typecast):
        return [arg]
    elif isinstance(arg, (list, tuple)):
        return list(arg)
    elif isinstance(arg, str):
        return [typecast(x.strip()) for x in arg.split(delimiter)]
    else:
        raise ValueError(f"cant parse into list: {typecast = } {arg = }")


def plot_loss(
    log_path: str,
    window_sizes: int | Sequence[int] | str = (10, 50, 100),
    raw_loss: bool | str = False,
):
    """
    Plots the loss over the training iteration, by reading log data from the log file specified by `log_path`.
    Optionally, the loss curve can be smoothed by computing a rolling average using the specified window size(s).
    Also, the raw loss curve can be plotted along with the smoothed curve.

    Parameters
    ----------
    log_path (str):
        Path to the training log file. This should be a file in `jsonl` format, as output by `training.train`.
    window_sizes (int or list[int] or str, optional):
        Window size(s) for computing rolling average of loss. Default is (10, 50, 100).
    raw_loss (bool or str, optional):
        Whether to plot the raw loss curve along with the smoothed curve. If a string is provided, it will be used as the format string for the raw loss curve. Default is False.

    Returns:
    None

    Raises:
    - ValueError: If `window_sizes` is not an int, list of ints, or a comma-separated string of ints.
    """
    data_raw: list[tuple] = gather_val(
        file=log_path,
        stream="train",
        keys=("n_sequences", "loss"),
    )

    data_config: list[dict] = gather_stream(
        file=log_path,
        stream="log_config",
    )

    print(f"{len(data_raw) = }")

    print(data_raw[:20])

    total_sequences, loss = zip(*data_raw)

    # compute a rolling average
    # this messy part here is just for handling command line arguments
    window_sizes = parse_argument_list(window_sizes, typecast=int)

    loss_rolling_arr: list[np.ndarray] = [
        rolling_average(loss, window)
        for window in window_sizes
        if window < len(loss)  # just skip windows that are too large
    ]

    if raw_loss:
        raw_loss_fmt: str = raw_loss if isinstance(raw_loss, str) else ","
        plt.plot(total_sequences, loss, raw_loss_fmt, label="raw losses")
    for cv, loss_rolling in zip(window_sizes, loss_rolling_arr):
        plt.plot(
            total_sequences[cv - 1 :],
            loss_rolling,
            "-",
            label=f"rolling avg $(\\pm {cv})$",
        )

    plt.ylabel("Loss")
    plt.xlabel("Total sequences")
    plt.yscale("log")
    title: str = ";  ".join(
        [
            f"dataset={get_any_from_stream(data_config, 'dataset_cfg')['name']}",
            f"train_config={get_any_from_stream(data_config, 'train_cfg')['name']}",
            f"lr={get_any_from_stream(data_config, 'train_cfg')['optimizer_kwargs']['lr']}",
            # vocab size is not in the model config -- this is just printing anyway
            # f"vocab_size={get_any_from_stream(data_config, 'model_cfg')['vocab_size']}",
        ]
    )
    plt.title(title)
    plt.legend()
    plt.show()


def rolling_average(x: Tuple[float], window: int):
    if window > len(x):
        raise ValueError(
            f"Window for rolling average {window} is greater than the number of datapoints {len(x)}"
        )

    return list(np.convolve(x, np.ones(window) / window, mode="valid"))
