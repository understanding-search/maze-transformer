import enum
from functools import cached_property
import json
from pathlib import Path
import typing
import warnings

import numpy as np
import torch
from muutils.json_serialize import (
    SerializableDataclass,
    serializable_dataclass,
    serializable_field,
    JSONitem,
)
from muutils.zanj import ZANJ
from muutils.tensor_utils import DTYPE_MAP, ATensor
from muutils.misc import sanitize_fname
from torch.utils.data import Dataset

from maze_transformer.utils.utils import get_device


def _dtype_serialization_fn(datatype: torch.dtype | np.dtype) -> str:
    """convert torch dtype to string, while checking that the conversion is reversible"""
    x_str: str = str(datatype)
    assert x_str in DTYPE_MAP, f"unknown dtype {datatype}"
    assert DTYPE_MAP[x_str] == datatype
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

    def to_fname(self) -> str:
        """convert config to a filename"""
        self_json_str: str = json.dumps(self.serialize())
        self_json_hash: int = int(abs(hash(self_json_str))%1e10)
        warnings.warn(f"using fallblack to_fname() method for {self.__class__.__name__}, this should be implemented by subclasses!")
        return sanitize_fname(f"f{self.name}_{self_json_hash}.zanj")


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

class SaveFormats(enum.Enum):
    OBJECTS: str = "objects"
    TOKENS: str = "tokens"
    ARRAY: str = "array"


class GPTDataset(Dataset):
    """wrapper for torch dataset with some extra functionality

    (meaning the functionality should be inherited in downstream classes)
    """

    @classmethod
    def from_config(
            cls, 
            cfg: GPTDatasetConfig,
            do_generate: bool = True,
            load_local: bool = True,
            save_local: bool = True,
            save_formats: set[SaveFormats] = {SaveFormats.OBJECTS, SaveFormats.TOKENS},
            do_download: bool = True,
            local_base_path: Path = Path("data/maze_dataset"),
            **kwargs,
        ) -> None:
        """base class for gpt datasets

        priority of loading:
        1. load from local
        2. download
        3. generate

        note: `GPTDatasetConfig` should implement a `to_fname` method that returns a unique filename for the config

        # Requires:
        the following methods should be implemented in subclasses:
         - `__init__(self, cfg: GPTDatasetConfig, **kwargs)`
            initialize the dataset from a given config. kwargs are not passed through, the kwargs should take the actual generated or loaded data (a list of objects or sequences probably)
         - `generate(cls, cfg: GPTDatasetConfig, **kwargs) -> GPTDataset`
            generate the dataset from a given config. kwargs are passed through from `from_config`, and should only contain things that dont belong in the config (i.e. how many threads to use for generation)
         - `serialize(self) -> JSONitem`
            serialize the dataset to a ZANJ-serializable object, including:
             - config
             - data in formats specified by `self.save_formats`
         - `load(cls, data: JSONitem) -> GPTDataset`
            load the dataset from a ZANJ-serializable object
         - `download(cls, cfg: GPTDatasetConfig, **kwargs) -> GPTDataset`
            given a config, try to download a dataset from some source. kwargs are passed through from `from_config`, and should only contain things that dont belong in the config (i.e. some kind of auth token or source url)
         - `__len__(self) -> int`
            return the length of the dataset, required for `torch.utils.data.Dataset`
         - `__getitem__(self, i: int) -> list[str]`
            return the ith item in the dataset, required for `torch.utils.data.Dataset`
         - `get(self, i: int, fmt: SaveFormats) -> Any`
            return the ith item in the dataset, required for `torch.utils.data.Dataset`
         - `get_all_lengths(self) -> list[int]`
            get the lengths of all sequences in the dataset

        # Parameters:
         - `cfg : GPTDatasetConfig`   
            config for the dataset, used to generate the dataset
         - `do_generate : bool`   
            whether to generate the dataset if it isn't found
            (defaults to `True`)
         - `load_local : bool`   
            whether to try finding the dataset locally
            (defaults to `True`)
         - `save_local : bool`   
            whether to save the dataset locally if it is generated or downloaded
            (defaults to `True`)
         - `save_formats : set[SaveFormats]`   
            which formats to save the dataset in
            (defaults to `{SaveFormats.OBJECTS, SaveFormats.TOKENS}`)
         - `do_download : bool`   
            whether to try downloading the dataset
            (defaults to `True`)
         - `local_base_path : Path`   
            where to save the dataset
            (defaults to `Path("data/maze_dataset")`)

        # Returns:
         - `GPTDataset`
            the dataset, as you wanted it
        
        # Implements:
         - `save(self, file_path: str) -> None`
            save the dataset to a file, using ZANJ
         - `read(cls, file_path: str) -> GPTDataset`
            read the dataset from a file, using ZANJ
         - `get_all(self, fmt: SaveFormats) -> Iterator[Any]`
            get all items in the dataset, in the specified format

        """

        local_base_path = Path(local_base_path)
        fname: str = cfg.to_fname()
        output: GPTDataset|None = None
        did_load_local: bool = False

        if not (load_local or do_download or do_generate):
            raise ValueError("no way to load dataset! you said not to load local, not to download, and not to generate")

        # try loading
        if load_local:
            if (local_base_path / fname).exists():
                output = cls.read(local_base_path / fname)
                did_load_local = True

        if do_download:
            try:
                output = cls.download(cfg, **kwargs)
            except NotImplementedError:
                pass

        if do_generate:
            output = cls.generate(cfg, **kwargs)

        # check and save
        if output is None:
            raise ValueError("failed to load dataset!")
        
        if output.cfg != cfg:
            raise ValueError(f"config mismatch: {cfg.diff(output.cfg)}")
        
        if save_local and not did_load_local:
            output.save(local_base_path / fname)

        return output

    def get_all(self, fmt: SaveFormats) -> typing.Iterator[typing.Any]:
        for idx in range(len(self)):
            yield self.get(idx, fmt)

    def save(self, file_path: str, zanj: ZANJ | None = None):
        if zanj is None:
            zanj = ZANJ()
        zanj.save(self.serialize(), file_path)

    @classmethod
    def read(cls, file_path: str, zanj: ZANJ | None = None) -> "GPTDataset":
        if zanj is None:
            zanj = ZANJ()
        return cls.load(zanj.read(file_path))

    def serialize(self) -> JSONitem:
        raise NotImplementedError()
    
    @classmethod
    def load(cls, data: JSONitem) -> "GPTDataset":
        raise NotImplementedError()

    @classmethod
    def generate(cls, cfg: GPTDatasetConfig, **kwargs) -> "GPTDataset":
        raise NotImplementedError()

    @classmethod
    def download(cls, cfg: GPTDatasetConfig, **kwargs) -> "GPTDataset":
        raise NotImplementedError()

    def get_all_lengths(self) -> list[int]:
        """get the lengths of all sequences"""
        raise NotImplementedError()