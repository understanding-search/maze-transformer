import functools
import multiprocessing
import typing
import warnings
from functools import cached_property
from typing import Callable
import copy

import numpy as np
from jaxtyping import Int
from muutils.json_serialize import JSONitem, serializable_dataclass, serializable_field
from muutils.json_serialize.util import safe_getsource, string_as_lines
from muutils.misc import sanitize_fname
import tqdm

from maze_transformer.generation.constants import SPECIAL_TOKENS, Coord, CoordArray, CoordTup
from maze_transformer.generation.generators import GENERATORS_MAP, LatticeMazeGenerators
from maze_transformer.generation.lattice_maze import LatticeMaze, SolvedMaze, TargetedLatticeMaze
from maze_transformer.training.dataset import (
    DatasetFilterProtocol,
    GPTDataset,
    GPTDatasetConfig,
    IndexedArray,
    SaveFormats,
    register_wrap_dataset_filter,
)
from maze_transformer.training.tokenizer import maze_to_tokens
from maze_transformer.utils.utils import register_method

_MAZEDATASET_PROPERTIES_TO_SERIALIZE: list[str] = [
    "padding_token_index",
    "token_arr",
    "tokenizer_map",
    "grid_shape",
    # "node_token_map", # doesn't work by default due to keys being tuples
    "token_node_map",
    "n_tokens",
]

# TODO: re-add later, depends on a feature coming in muutils 0.3.2
__MAZEDATASET_PROPERTIES_TO_VALIDATE: list[str] = [
    "token_arr",
    "padding_token_index",
    "tokenizer_map",
    "grid_shape",
    "token_node_map",
    "n_tokens",
]


def _load_maze_ctor(maze_ctor_serialized: str | dict) -> Callable:
    if isinstance(maze_ctor_serialized, dict):
        # this is both the new and old version of the serialization
        return GENERATORS_MAP[maze_ctor_serialized["__name__"]]
    elif isinstance(maze_ctor_serialized, str):
        # this is a version I switched to for a while but now we are switching back
        warnings.warn(
            f"you are loading an old model/config!!! this should not be happening, please report to miv@knc.ai"
        )
        return GENERATORS_MAP[maze_ctor_serialized]
    else:
        raise ValueError(
            f"maze_ctor_serialized is of type {type(maze_ctor_serialized)}, expected str or dict"
        )


@serializable_dataclass(
    kw_only=True, properties_to_serialize=_MAZEDATASET_PROPERTIES_TO_SERIALIZE
)
class MazeDatasetConfig(GPTDatasetConfig):
    """maze dataset configuration, including tokenizers"""

    grid_n: int
    n_mazes: int
    maze_ctor: Callable = serializable_field(
        default_factory=lambda: LatticeMazeGenerators.gen_dfs,
        serialization_fn=lambda gen_func: {
            "__name__": gen_func.__name__,
            "__module__": gen_func.__module__,
            "__doc__": string_as_lines(gen_func.__doc__),
            "source_code": safe_getsource(gen_func),
        },
        loading_fn=lambda data: _load_maze_ctor(data["maze_ctor"]),
    )

    # paths_per_maze: int = 5,
    # p_min_tgt_dist: float = 0.2,

    @property
    def grid_shape(self) -> CoordTup:
        return (self.grid_n, self.grid_n)
    
    @property
    def grid_shape_np(self) -> Coord:
        return np.array(self.grid_shape)

    @cached_property
    def node_token_map(self) -> dict[CoordTup, str]:
        """map from node to token"""
        return {tuple(c): f"({c[0]},{c[1]})" for c in np.ndindex(self.grid_shape)}

    @cached_property
    def token_node_map(self) -> dict[str, CoordTup]:
        """map from token to node"""
        return {v: k for k, v in self.node_token_map.items()}

    @cached_property
    def token_arr(self) -> list[str]:
        """map from index to token"""
        return [
            *list(SPECIAL_TOKENS.values()),
            *list(self.node_token_map.values()),
        ]

    @property
    def n_tokens(self) -> int:
        return len(self.token_arr)

    @cached_property
    def padding_token_index(self) -> str:
        return self.tokenizer_map[SPECIAL_TOKENS["padding"]]

    def to_fname(self) -> str:
        # self_json_str: str = json.dumps(self.serialize())
        # self_json_hash: int = int(abs(hash(self_json_str))%1e5)
        # return sanitize_fname(f"{self.name}-g{self.grid_n}-n{self.n_mazes}-h{self_json_hash}")
        return sanitize_fname(
            f"{self.name}-g{self.grid_n}-n{self.n_mazes}-a_{self.maze_ctor.__name__.removeprefix('gen_')}"
        )


def _generate_maze_helper(positions: CoordArray) -> SolvedMaze:
    maze: LatticeMaze = _GLOBAL_WORKER_CONFIG.maze_ctor(grid_shape=_GLOBAL_WORKER_CONFIG.grid_shape_np)
    return SolvedMaze.from_lattice_maze(
        lattice_maze=maze, 
        solution=np.array(maze.find_shortest_path(positions[0], positions[1])),
    )

def _maze_gen_init_worker(config: MazeDatasetConfig):
    global _GLOBAL_WORKER_CONFIG
    _GLOBAL_WORKER_CONFIG = config

class MazeDataset(GPTDataset):
    """maze dataset"""

    def __init__(
        self,
        cfg: MazeDatasetConfig,
        mazes: typing.Sequence[SolvedMaze],
    ) -> None:
        super().__init__()
        self.cfg: MazeDatasetConfig = cfg
        self.mazes: list[SolvedMaze] = list(mazes)

    def data_hash(self) -> int:
        return hash(tuple(self.mazes))

    def get(
        self, index: int, fmt: SaveFormats = SaveFormats.OBJECTS
    ) -> SolvedMaze | list[str] | np.ndarray:
        """get a single maze, as one of the formats"""
        if fmt == SaveFormats.OBJECTS:
            return self.mazes[index]
        elif fmt == SaveFormats.TOKENS:
            return maze_to_tokens(self.mazes[index], self.cfg.node_token_map)
        elif fmt == SaveFormats.ARRAY:
            raise NotImplementedError("getting as array not implemented yet")
        else:
            raise ValueError(
                f"unknown fmt {fmt}, expected an instance of `SaveFormats` enum"
            )

    def __getitem__(self, index: int, pad: bool = True) -> str:
        """index into mazes_array.arr, getting from the start of the correct sequence, padding if necessary"""

        return " ".join(maze_to_tokens(self.mazes[index], self.cfg.node_token_map))
        # tokens: list[str] = maze_to_tokens(self.mazes[index], self.cfg.node_token_map)

        # if pad:
        #     remaining_len: int = self.cfg.seq_len_max - len(tokens)
        #     return (
        #         [self.cfg.token_arr[self.cfg.padding_token_index] for _ in range(remaining_len)]
        #         + tokens
        #     )
        # else:
        #     return tokens

    mazes_objs: list[SolvedMaze] = cached_property(
        lambda self: list(self.get_all(fmt=SaveFormats.OBJECTS))
    )
    mazes_tokens: list[list[str]] = cached_property(
        lambda self: list(self.get_all(fmt=SaveFormats.TOKENS))
    )
    mazes_array: IndexedArray = cached_property(
        lambda self: IndexedArray(self.get_all(fmt=SaveFormats.ARRAY))
    )

    def __len__(self) -> int:
        return len(self.mazes)

    def get_all_lengths(self) -> list[int]:
        raise NotImplementedError()
        # return [len(m) for m in self.mazes_tokens]

    @classmethod
    def generate(
            cls, 
            cfg: MazeDatasetConfig,
            do_parallel: bool = True,
            verbose: bool = False,
        ) -> "MazeDataset":
        mazes: list[SolvedMaze] = list()
        endpoint_nodes: Int[np.int8, "maze_index 2 2"] = np.random.randint(
            0,
            cfg.grid_shape,
            (cfg.n_mazes, 2, 2),
        )
        # TODO: filter min distanced based on MazeDatasetConfig

        solved_mazes: list[SolvedMaze]
        tqdm_kwargs: dict = dict(
            total=cfg.n_mazes,
            unit="maze",
            desc="generating & solving mazes",
            disable=not verbose,
        )
        if do_parallel:
            with multiprocessing.Pool(initializer=_maze_gen_init_worker, initargs=(cfg,)) as pool:
                solved_mazes = list(
                    tqdm.tqdm(
                        pool.imap(
                            _generate_maze_helper,
                            endpoint_nodes,
                        ),
                        **tqdm_kwargs,
                    )
                )
        else:
            _maze_gen_init_worker(cfg)
            solved_mazes = list(
                tqdm.tqdm(
                    map(
                        _generate_maze_helper,
                        endpoint_nodes,
                    ),
                    **tqdm_kwargs,
                )
            )

        return cls(
            cfg=cfg,
            mazes=solved_mazes,
        )

    @classmethod
    def download(cls, cfg: MazeDatasetConfig, **kwargs) -> "MazeDataset":
        raise NotImplementedError("not implemented yet")

    @classmethod
    def load(cls, data: JSONitem) -> "MazeDataset":
        """load from zanj/json"""
        assert data["__format__"] == "MazeDataset"
        return cls(
            cfg=MazeDatasetConfig.load(data["cfg"]),
            mazes=[SolvedMaze.load(m) for m in data["mazes"]],
        )

    def serialize(self) -> JSONitem:
        """serialize to zanj/json"""
        return {
            "__format__": "MazeDataset",
            "cfg": self.cfg.serialize(),
            "mazes": [m.serialize() for m in self.mazes],
        }

    @classmethod
    def disk_load(cls, path: str, **kwargs) -> "MazeDataset":
        """load from disk"""
        warnings.warn(
            "deprecated, use `MazeDataset.read(path)` or `MazeDataset.load(ZANJ().read(path)))` instead",
            DeprecationWarning,
        )
        if kwargs:
            warnings.warn(
                f"kwargs to disk_load dont do anything: {kwargs = }", DeprecationWarning
            )
        return cls.read(path)


MazeDatasetConfig._dataset_class = MazeDataset


MAZE_DATASET_CONFIGS: dict[str, MazeDatasetConfig] = {
    cfg.to_fname(): cfg
    for cfg in [
        MazeDatasetConfig(
            name="test",
            grid_n=3,
            n_mazes=5,
            maze_ctor=LatticeMazeGenerators.gen_dfs,
        ),
    ]
}



MAZE_DATASET_FILTERS: dict[str, DatasetFilterProtocol] = dict()


class MazeDatasetFilters:

    @register_wrap_dataset_filter(MAZE_DATASET_FILTERS)    
    def path_length(dataset: MazeDataset, min_length: int) -> MazeDataset:
        return MazeDataset(
            cfg=dataset.cfg,
            mazes=list(filter(lambda m: len(m.path) >= min_length, dataset.mazes)),
        )


