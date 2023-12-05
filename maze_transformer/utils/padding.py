from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Int
from muutils.mlutils import chunks


def pad_and_batch_tensors(
    contexts_tokens: list[list[int]],
    batch_size: int,
    padding_idx: int,
    padding_dir: Literal["left", "right"],
    min_len: int = 0,
    max_len: int | None = None,
) -> list[Int[torch.Tensor, "batch pos"]]:
    """Pad and stack the tensors"""

    assert padding_dir in [
        "left",
        "right",
    ], f"padding_dir must be one of 'left' or 'right', got '{padding_dir}'"

    # Calculate the maximum sequence length across all contexts_tokens
    global_max_length = max(max(len(x) for x in contexts_tokens), min_len)

    if max_len is not None and global_max_length > max_len:
        raise ValueError(
            f"Sequence length exceeds the maximum allowed length: {max_len}"
        )

    contexts_tensored: list[Int[torch.Tensor, "batch pos"]] = []
    batch: list[list[int]]
    for batch in chunks(contexts_tokens, batch_size):
        batch_max_len: int = max(len(x) for x in batch)
        padded_batch = [
            F.pad(
                torch.tensor(x, dtype=torch.long),
                (
                    (batch_max_len - len(x), 0)
                    if padding_dir == "left"
                    else (0, global_max_length - len(x))
                ),
                value=padding_idx,
            )
            for x in batch
        ]

        # Stack the padded tensors
        batch_tensor = torch.stack(padded_batch)
        contexts_tensored.append(batch_tensor)

    return contexts_tensored
