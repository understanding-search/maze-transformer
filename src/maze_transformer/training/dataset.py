import json
import os
from pathlib import Path
import sys
import inspect
from functools import cached_property, partial
from itertools import chain, product
from typing import Any, Callable, Generic, Literal, NamedTuple, Sequence, TypeVar, Union
from dataclasses import dataclass, field
import multiprocessing

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import OpenAIGPTConfig
from tqdm import tqdm
from muutils.tensor_utils import ATensor, NDArray, DTYPE_MAP, lpad_array
from muutils.json_serialize import (
    json_serialize,
    dataclass_serializer_factory,
    dataclass_loader_factory,
    try_catch,
    JSONitem,
)
from muutils.misc import freeze
from muutils.statcounter import StatCounter

from maze_transformer.generation.latticemaze import (
    LatticeMaze,
    Coord,
    CoordTup,
    CoordArray,
)
from maze_transformer.generation.generators import LatticeMazeGenerators, GENERATORS_MAP
from maze_transformer.training.tokenizer import SPECIAL_TOKENS, MazeTokenizer


@dataclass(kw_only=True)
class GPTDatasetConfig:
    """base config class"""

    name: str
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    dtype: torch.dtype | np.dtype = field(default_factory=lambda: torch.int16)
    seq_len_min: int = 1
    seq_len_max: int = 512

    @cached_property
    def token_arr(self) -> list[str]:
        raise NotImplementedError()

    @cached_property
    def padding_token_idx(self) -> str:
        raise NotImplementedError()

    @cached_property
    def tokenizer_map(self) -> dict[str, int]:
        """map from token to index"""
        return {t: i for i, t in enumerate(self.token_arr)}

    @classmethod
    @property
    def _dataset_class(cls) -> type:
        raise NotImplementedError("this should be implemented by subclasses!")

    @cached_property
    def gpt_config_kwargs(self) -> dict:
        """gpt model config with vocab size, context size, and padding token"""
        return dict(
            vocab_size=len(self.token_arr),
            n_positions=self.seq_len_max,
            pad_token_id=self.padding_token_idx,  # The id of the _padding_ token.
            bos_token_id=self.padding_token_idx,  # The id of the _beginning-of-stream_ token.
            eos_token_id=self.padding_token_idx,  # The id of the _end-of-stream_ token.
        )

    def tokenize_seq(self, seq: list[str]) -> ATensor:
        """tokenize sequence"""
        return torch.tensor(
            [self.tokenizer_map[t] for t in seq],
            dtype=self.dtype,
            device="cpu",
        )

    def serialize(self) -> dict:
        raise NotImplementedError()

    @classmethod
    def load(cls, data: dict) -> "DatasetConfig":
        raise NotImplementedError()


@dataclass(kw_only=True)
class IndexedArray:
    """join a list of arrays into a single big one with indices

    mainly for allowing __getitem__ to work nice for datasets"""

    arr: ATensor
    idxs: ATensor

    def get_len(self, idx: int) -> int:
        return self.idxs[idx + 1] - self.idxs[idx]

    def get_all_lengths(self) -> ATensor:
        return torch.cat(
            [
                self.idxs[1:] - self.idxs[:-1],
                torch.tensor(
                    [self.arr.shape[0] - self.idxs[-1]],
                    dtype=self.idxs.dtype,
                    device=self.idxs.device,
                ),
            ]
        )

    @classmethod
    def from_sequences(cls, data: list[ATensor[("tokens")]]) -> "IndexedArray":
        """process many sequences into a single array, keeping track of sequence start indices

        example:
        f( [[a,b,c], [d,e]] ) -> IndexedArray(
                arr = [a,b,c,d,e],
                idxs = [0,3],
        )
        """
        arr: ATensor = torch.cat(data)
        idxs: ATensor = torch.cumsum(torch.tensor([0, *map(len, data)]), dim=0)[:-1]
        return cls(arr=arr, idxs=idxs)


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
