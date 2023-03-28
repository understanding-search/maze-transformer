from functools import cached_property

import numpy as np
import torch
from muutils.json_serialize import (
    SerializableDataclass,
    serializable_dataclass,
    serializable_field,
)
from muutils.tensor_utils import DTYPE_MAP, ATensor
from torch.utils.data import Dataset

from maze_transformer.utils.utils import get_device


def _dtype_serialization_fn(x: torch.dtype | np.dtype) -> str:
    x_str: str = str(x)
    assert x_str in DTYPE_MAP, f"unknown dtype {x}"
    assert DTYPE_MAP[x_str] == x
    return x_str


@serializable_dataclass(
    kw_only=True,
    properties_to_serialize=["token_arr", "padding_token_index", "tokenizer_map"],
)
class GPTDatasetConfig(SerializableDataclass):
    """base config class"""

    name: str
    device: torch.device = serializable_field(
        default_factory=lambda: torch.device(get_device()),
        serialization_fn=lambda x: str(x),
        loading_fn=lambda data: torch.device(data["device"]),
    )
    dtype: torch.dtype | np.dtype = serializable_field(
        default_factory=lambda: torch.int16,
        serialization_fn=_dtype_serialization_fn,
        loading_fn=lambda data: DTYPE_MAP[data["dtype"]],
    )
    seq_len_min: int = serializable_field(default=1)
    seq_len_max: int = serializable_field(default=512)

    @cached_property
    def token_arr(self) -> list[str]:
        raise NotImplementedError()

    @cached_property
    def padding_token_index(self) -> str:
        raise NotImplementedError()

    @cached_property
    def tokenizer_map(self) -> dict[str, int]:
        """map from token to index"""
        return {t: i for i, t in enumerate(self.token_arr)}

    @classmethod
    @property
    def _dataset_class(cls) -> type:
        raise NotImplementedError("this should be implemented by subclasses!")

    def tokenize_seq(self, seq: list[str]) -> ATensor:
        """tokenize sequence"""
        return torch.tensor(
            [self.tokenizer_map[t] for t in seq],
            dtype=self.dtype,
            device="cpu",
        )


@serializable_dataclass(kw_only=True)
class IndexedArray(SerializableDataclass):
    """Contains a tensor made by concatenating a list of tensors, and a second tensor indicating the starting indices
    of the original tensors in the first one. Mainly for getting __getitem__ to work nicely with datasets

    arr: tensor containing all the elements of the original arrays: [1, 2], [3, 4] -> [1, 2, 3, 4]
    indices: tensor indicating the starting index in arr of each original array: [1, 2], [3, 4] -> [0, 2]
    """

    arr: torch.Tensor
    indices: torch.Tensor

    def get_len(self, i: int) -> int:
        if i + 1 < len(self.indices):
            return self.indices[i + 1] - self.indices[i]

        return self.arr.size(0) - self.indices[i]

    def get_all_lengths(self) -> torch.Tensor:
        return torch.cat(
            [
                self.indices[1:] - self.indices[:-1],
                torch.tensor(
                    [self.arr.size(0) - self.indices[-1]],
                    dtype=self.indices.dtype,
                    device=self.indices.device,
                ),
            ]
        )

    @classmethod
    def from_sequences(cls, data: list[ATensor[("tokens")]]) -> "IndexedArray":
        """Process many sequences into a single array, keeping track of sequence start indices

        example:
        f( [[a,b,c], [d,e]] ) -> IndexedArray(
                arr = [a,b,c,d,e],
                indices = [0,3]
        )
        """

        arr: ATensor = torch.cat(data)
        indices: ATensor = torch.cumsum(torch.tensor([0, *map(len, data)]), dim=0)[:-1]
        return cls(arr=arr, indices=indices)


class GPTDataset(Dataset):
    """wrapper for torch dataset with some extra functionality

    (meaning the functionality should be inherited in downstream classes)
    """

    def get_all_lengths(self) -> list[int]:
        """get the lengths of all sequences"""
        raise NotImplementedError()

    @classmethod
    def config_save_name(cls) -> str:
        """name of the config file"""
        raise NotImplementedError()
