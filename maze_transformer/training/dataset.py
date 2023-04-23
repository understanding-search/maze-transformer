import copy
import enum
import functools
import json
import types
import typing
import warnings
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from muutils.json_serialize import (
    JSONitem,
    SerializableDataclass,
    serializable_dataclass,
    serializable_field,
)
from muutils.misc import sanitize_fname
from muutils.tensor_utils import DTYPE_MAP, ATensor
from muutils.zanj import ZANJ
from torch.utils.data import Dataset

from maze_transformer.utils.utils import (
    DEFAULT_SEED,
    GLOBAL_SEED,
    get_device,
    set_reproducibility,
)


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
    """base GPTDatasetConfig class"""

    name: str
    # TODO: get rid of device here? belongs in training config
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
    seed: int | None = serializable_field(default=DEFAULT_SEED)
    applied_filters: list[
        dict[typing.Literal["name", "kwargs"], str | dict]
    ] = serializable_field(default_factory=list)

    def __post_init__(self):
        assert self.seq_len_min <= self.seq_len_max
        # if seed set to None, then generate a new random seed
        if self.seed is None:
            self.seed = torch.random.seed() % 2**31

        if (DEFAULT_SEED != self.seed) and (GLOBAL_SEED != self.seed):
            warnings.warn(
                f"in GPTDatasetConfig {self.name=}, {self.seed=} is trying to override {GLOBAL_SEED=} which has already been changed elsewhere from {DEFAULT_SEED=}"
            )

        set_reproducibility(self.seed)

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
        self_json_hash: int = int(abs(hash(self_json_str)) % 1e10)
        warnings.warn(
            f"using fallblack to_fname() method for {self.__class__.__name__}, this should be implemented by subclasses!"
        )
        return sanitize_fname(f"f{self.name}_{self_json_hash}")


# TODO: This class is not referenced anywhere. Can we delete?
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

    _FILTER_NAMESPACE: type = "this isn't a filter namespace! you have to initialize this by registering with `register_filter_namespace_for_dataset`"  # type: ignore

    @classmethod
    def from_config(
        cls,
        cfg: GPTDatasetConfig,
        do_generate: bool = True,
        load_local: bool = True,
        save_local: bool = True,
        save_formats: set[SaveFormats] = {SaveFormats.OBJECTS, SaveFormats.TOKENS},
        zanj: ZANJ | None = None,
        do_download: bool = True,
        local_base_path: Path = Path("data/maze_dataset"),
        **kwargs,
    ) -> "GPTDataset":
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
         -  `update_self_config(self) -> None`
            update the config of the dataset to match the current state of the dataset, used primarily in filtering and validation
         -  decorating the appropriate filter namespace with `register_filter_namespace_for_dataset(your_dataset_class)` if you want to use filters

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
         - `filter_by(self)`
            returns a namespace class
         -  `_filter_namespace(self) -> Class`
            returns a namespace class for filtering the dataset, checking that method
         - `_apply_filters_from_config(self) -> None`
            apply filters to the dataset, as specified in the config. used in `from_config()` but only when generating

        """

        local_base_path = Path(local_base_path)
        fname: Path = Path(f"{cfg.to_fname()}.zanj")
        output: GPTDataset | None = None
        did_load_local: bool = False
        if zanj is None:
            zanj = ZANJ()

        if not (load_local or do_download or do_generate):
            raise ValueError(
                "no way to load dataset! you said not to load local, not to download, and not to generate"
            )

        # try loading
        if load_local:
            if (local_base_path / fname).exists():
                output = cls.read(local_base_path / fname, zanj=zanj)
                did_load_local = True

        if do_download:
            try:
                output = cls.download(cfg, **kwargs)
            except NotImplementedError:
                pass

        if do_generate:
            output = cls.generate(cfg, **kwargs)
            # only if we generated it, apply filters
            output = output._apply_filters_from_config()

        # check and save
        if output is None:
            raise ValueError("failed to load dataset!")

        if output.cfg != cfg:
            raise ValueError(f"config mismatch: {cfg.diff(output.cfg)}")

        if save_local and not did_load_local:
            output.save(local_base_path / fname, zanj=zanj)

        return output

    # getting data
    def get_all(self, fmt: SaveFormats) -> typing.Iterator[typing.Any]:
        for idx in range(len(self)):
            yield self.get(idx, fmt)

    def get_all_lengths(self) -> list[int]:
        """get the lengths of all sequences"""
        raise NotImplementedError()

    def save(self, file_path: str, zanj: ZANJ | None = None):
        if zanj is None:
            zanj = ZANJ()
        zanj.save(self.serialize(), file_path)

    # serialization & loading
    @classmethod
    def read(cls, file_path: str, zanj: ZANJ | None = None) -> "GPTDataset":
        if zanj is None:
            zanj = ZANJ()
        return cls.load(zanj.read(file_path))

    def serialize(self) -> JSONitem:
        raise NotImplementedError()

    def data_hash(self) -> int:
        raise NotImplementedError()

    @classmethod
    def load(cls, data: JSONitem) -> "GPTDataset":
        raise NotImplementedError()

    # generating & downloading
    @classmethod
    def generate(cls, cfg: GPTDatasetConfig, **kwargs) -> "GPTDataset":
        raise NotImplementedError()

    @classmethod
    def download(cls, cfg: GPTDatasetConfig, **kwargs) -> "GPTDataset":
        raise NotImplementedError()

    # filtering
    def update_self_config(self):
        """update the config of the dataset to match the actual data, if needed

        for example, adjust number of mazes after filtering
        """
        pass

    class FilterBy:
        """thanks GPT-4"""

        def __init__(self, dataset: "GPTDataset"):
            self.dataset: "GPTDataset" = dataset

        def __getattr__(self, name: str) -> typing.Callable[..., "GPTDataset"]:
            filter_func: DatasetFilterProtocol = getattr(
                self.dataset._FILTER_NAMESPACE, name
            )

            def wrapped_filter_func(**kwargs):
                return filter_func(self.dataset, **kwargs)

            return wrapped_filter_func

    @property
    def filter_by(self) -> "FilterBy":
        return self.FilterBy(self)

    def _apply_filters_from_config(self):
        """apply filters to the dataset, as specified in the config. used in `from_config()`"""
        output: GPTDataset = self
        # copy the list, and then clear it in the config
        applied_filters_old: list[
            dict[typing.Literal["name", "kwargs"], typing.Any]
        ] = copy.deepcopy(self.cfg.applied_filters)
        self.cfg.applied_filters = list()
        # apply the filters
        for filter_info in applied_filters_old:
            filter_name: str = filter_info["name"]
            if filter_name not in self._FILTER_NAMESPACE.__dict__:
                if filter_name.startswith("__custom__:"):
                    raise ValueError(
                        f"the dataset {self.cfg.to_fname()} was filtering using a custom filter: '{filter_name}', which we don't know about. add it to MazeDatasetFilters"
                    )
                else:
                    raise ValueError(
                        f"the dataset {self.cfg.to_fname()} was filtering using an unknown filter: '{filter_name}'"
                    )
            filter_kwargs: dict = filter_info["kwargs"]
            output = getattr(self.filter_by, filter_name)(**filter_kwargs)
        # update the config
        self.update_self_config()
        assert (
            self.cfg.applied_filters == applied_filters_old
        ), f"config mismatch: {self.cfg.applied_filters} != {applied_filters_old}"
        return output


def register_filter_namespace_for_dataset(dataset_cls: type[GPTDataset]) -> type:
    """register the namespace class with the given dataset class

    sets:
    ```python
    dataset_cls._FILTER_NAMESPACE = filter_namespace_cls
    filter_namespace_cls._BASE_DATASET = dataset_cls
    ```
    and also adds a `_filter_namespace_cls` attribute to every static method in the namespace class for use in `register_wrap_dataset_filter`
    """

    def decorator(filter_namespace_cls: type) -> type:
        dataset_cls._FILTER_NAMESPACE = filter_namespace_cls
        filter_namespace_cls._BASE_DATASET = dataset_cls

        # TODO: remove this? it's only useful for checking
        for method_name, method in filter_namespace_cls.__dict__.items():
            if (
                isinstance(method, typing.Callable)
                and isinstance(method, types.FunctionType)
                and not method.__name__.startswith("_")
            ):
                setattr(method, "_filter_namespace_cls", filter_namespace_cls)

        return filter_namespace_cls

    return decorator


class DatasetFilterProtocol(typing.Protocol):
    def __call__(
        self,
        dataset: GPTDataset,
        **kwargs,
    ) -> GPTDataset:
        ...


def register_wrap_dataset_filter(
    method: DatasetFilterProtocol,
) -> DatasetFilterProtocol:
    """register a dataset filter, copying the underlying dataset and updating the config

    method should be a staticmethod of a namespace class registered with `register_filter_namespace_for_dataset`
    """

    @functools.wraps(method)
    def wrapper(dataset: GPTDataset, **kwargs):
        # copy and filter
        new_dataset: GPTDataset = copy.deepcopy(dataset)
        new_dataset = method(dataset, **kwargs)
        # update the config
        new_dataset.cfg.applied_filters.append(
            dict(name=method.__name__, kwargs=kwargs)
        )
        new_dataset.update_self_config()
        return new_dataset

    return wrapper
