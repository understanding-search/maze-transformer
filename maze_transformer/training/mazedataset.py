import json
import multiprocessing
import os
from functools import cached_property, partial
from typing import Callable

import numpy as np
import torch
from muutils.json_serialize import (
    JSONitem,
    json_serialize,
    serializable_dataclass,
    serializable_field,
)
from muutils.misc import freeze
from muutils.statcounter import StatCounter
from muutils.tensor_utils import ATensor, NDArray
from tqdm import tqdm

from maze_transformer.generation.generators import GENERATORS_MAP, LatticeMazeGenerators
from maze_transformer.generation.latticemaze import (
    CoordArray,
    CoordTup,
    LatticeMaze,
    SolvedMaze,
)
from maze_transformer.training.dataset import GPTDataset, GPTDatasetConfig, IndexedArray
from maze_transformer.training.tokenizer import SPECIAL_TOKENS, maze_to_tokens

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


@serializable_dataclass(kw_only=True, properties_to_serialize=_MAZEDATASET_PROPERTIES_TO_SERIALIZE)
class MazeDatasetConfig(GPTDatasetConfig):
    """maze dataset configuration, including tokenizers"""

    grid_n: int
    n_mazes: int
    maze_ctor: Callable = serializable_field(
        default_factory=lambda: LatticeMazeGenerators.gen_dfs,
        serialization_fn=lambda x: x.__name__,
        loading_fn=lambda data: GENERATORS_MAP[data["maze_ctor"]],
    )

    # paths_per_maze: int = 5,
    # p_min_tgt_dist: float = 0.2,

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.grid_n, self.grid_n)

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


class MazeDataset(GPTDataset):
    """maze dataset"""

    def __init__(
        self,
        cfg: MazeDatasetConfig,
        mazes_objs: list[SolvedMaze] | None = None,
        mazes_tokens: list[list[str]] | None = None,
        mazes_array: IndexedArray | None = None,
        paths: dict[str, str] = None,
    ) -> None:
        super().__init__()

        self.cfg: MazeDatasetConfig = cfg

        # get mode
        if (
            sum(1 if x is None else 0 for x in [mazes_objs, mazes_tokens, mazes_array])
            < 1
        ):
            raise ValueError(
                "at least one of mazes_objs, mazes_tokens, mazes_tokenized must be provided to MazeDataset"
            )

        # transfer
        self.mazes_objs: list[SolvedMaze] | None = mazes_objs
        self.mazes_tokens: list[list[str]] | None = mazes_tokens
        self.mazes_array: IndexedArray | None = mazes_array

        # process into tokens
        if (self.mazes_objs is not None) and (self.mazes_tokens is None):
            with multiprocessing.Pool() as pool:
                self.mazes_tokens = list(
                    tqdm(
                        pool.starmap(
                            partial(maze_to_tokens, node_token_map=cfg.node_token_map),
                            self.mazes_objs,
                        ),
                        total=len(self.mazes_objs),
                        desc="tokenizing mazes",
                        unit="maze",
                    )
                )

        # process tokens into tokenized
        if (self.mazes_tokens is not None) and (mazes_array is None):
            max_len: int = max(len(t) for t in self.mazes_tokens)
            if max_len > cfg.seq_len_max:
                raise ValueError(f"{max_len=} exceeds {cfg.seq_len_max=}")

            self.mazes_array = IndexedArray.from_sequences(
                [cfg.tokenize_seq(m) for m in self.mazes_tokens]
            )

        # validate
        # if any(x is None for x in (self.mazes_objs, self.mazes_tokens, self.mazes_tokenized)):
        # 	raise ValueError(f"MazeDataset invalid, something is None: {type(self.mazes_objs) = } {type(self.mazes_tokens) = } {type(self.mazes_tokenized) = }")

        if self.mazes_objs is not None and self.mazes_tokens is not None:
            if len(self.mazes_objs) != len(self.mazes_tokens):
                raise ValueError(
                    f"MazeDataset invalid: {len(self.mazes_objs) = }, {len(self.mazes_tokens) = }"
                )

        if self.mazes_objs is not None and self.mazes_array.indices is not None:
            if len(self.mazes_objs) != len(self.mazes_array.indices):
                raise ValueError(
                    f"MazeDataset invalid: {len(self.mazes_objs) = }, {len(self.mazes_array.indices) = }"
                )

        if self.mazes_tokens is not None and self.mazes_array.indices is not None:
            if len(self.mazes_tokens) != len(self.mazes_array.indices):
                raise ValueError(
                    f"MazeDataset invalid: {len(self.mazes_tokens) = }, {len(self.mazes_array.indices) = }"
                )

    def __getitem__(self, index: int) -> ATensor[("tokens")]:
        """index into mazes_array.arr, getting from the start of the correct sequence, padding if necessary"""

        # A nice-to-have refactor would be to have some notion of a minimum sequence length here, such that this method
        # never returns a sequence below that length. The motivation here is that for a particular dataset we might know
        # that the first N tokens are always part of the maze, so this method can safetly skip that many before it finds
        # the start of the correct sequence. The min value could be part of the dataset config.

        # last element in mazes_array.indices whose value is smaller than `index`
        sequence_index: int = torch.searchsorted(self.mazes_array.indices, index) - 1
        # slice the array from the start of the sequence to `index`, including `index`
        end_arr_index: int = min(
            index + 1,  # up to end of sequence
            self.mazes_array.indices[sequence_index]
            + self.cfg.seq_len_max,  # up to sequence length cutoff
        )
        subseq: ATensor = self.mazes_array.arr[
            self.mazes_array.indices[sequence_index] : end_arr_index
        ]
        # left-pad the sequence
        return torch.nn.functional.pad(
            subseq,
            (self.cfg.seq_len_max + 1 - len(subseq), 0),
            value=self.cfg.padding_token_index,
        )

    def __len__(self) -> int:
        return len(self.mazes_array.arr)

    def get_all_lengths(self) -> list[int]:
        return self.mazes_array.get_all_lengths().tolist()

    @classmethod
    def gen_default(
        cls,
        cfg: MazeDatasetConfig,
    ) -> "MazeDataset":
        """generate a dataset of mazes

        p_min_tgt_dist is the minimum manhattan distance between the start and target,
        as a fraction of max of the maze's dimensions
        """

        # Currently we don't enforce a minimum distance between the start and end of the path. If we need to do this in future, there's a beginning of an implementation here:
        # n_min_tgt_dist: int = int(max(maze.grid_shape) * p_min_tgt_dist)

        """if np.abs(start_node - end_node).sum() < n_min_tgt_dist:
			# if too close, move end node towards the corner opposite the start node
			opposite_corner: CoordTup = (
				maze.grid_shape[0] * round(start_node[0] / maze.grid_shape[0]),
				maze.grid_shape[1] * round(start_node[1] / maze.grid_shape[1]),
			)
			# end_node +=
		"""

        solved_mazes: list[SolvedMaze] = list()
        endpoint_nodes: NDArray[
            (("maze_index", cfg.n_mazes), ("start_end", 2), ("coord", 2)), np.int8
        ] = np.random.randint(0, cfg.grid_shape, (cfg.n_mazes, 2, 2))

        print(endpoint_nodes)

        for i, (c_start, c_end) in enumerate(endpoint_nodes):
            m: LatticeMaze = cfg.maze_ctor(cfg.grid_shape)
            path: CoordArray = np.array(m.find_shortest_path(c_start, c_end))
            solved_mazes.append(SolvedMaze(m, path))

        return cls(
            cfg=cfg,
            mazes_objs=solved_mazes,
        )

    def serialize_config(self) -> JSONitem:
        return json_serialize(self.cfg)

    @freeze
    class DISK_SAVE_FILES:
        """namespace for filenames"""

        cfg: str = "cfg.json"
        # Not implemented
        # obj: str = "maze_obj.jsonl"
        tokens: str = "maze_tokens.jsonl"
        tokenized: str = "maze_tokenized.npz"

    @classmethod
    def config_save_name(cls) -> str:
        return cls.DISK_SAVE_FILES.cfg

    def disk_save(
        self,
        path_base: str = "data/test-001",
        do_config: bool = True,
        do_obj: bool = False,
        do_tokens: bool = True,
        do_tokenized: bool = True,
    ) -> None:
        # make the appropriate directories
        print(f"saving to '{path_base}'")
        os.makedirs(path_base, exist_ok=False)

        if do_config:
            # save config as json, with metadata
            config_out: dict[str, JSONitem] = {
                **json_serialize(self.cfg),
                "_postgen_meta": {
                    "seq_len_stats": StatCounter(
                        self.mazes_array.get_all_lengths().tolist()
                    ).summary(),
                },
            }
            with open(f"{path_base}/{self.DISK_SAVE_FILES.cfg}", "w") as f:
                json.dump(config_out, f, indent="\t")

        if do_obj:
            raise NotImplementedError("do_obj not implemented")

        if do_tokens:
            # save tokens as jsonl
            with open(f"{path_base}/{self.DISK_SAVE_FILES.tokens}", "w") as f:
                for x in self.mazes_tokens:
                    f.write(" ".join(x) + "\n")

        if do_tokenized:
            # save tokenized data as npz
            np.savez(
                f"{path_base}/{self.DISK_SAVE_FILES.tokenized}",
                **dict(
                    arr=self.mazes_array.arr.cpu().numpy(),
                    indices=self.mazes_array.indices.cpu().numpy(),
                ),
            )

    @classmethod
    def disk_load(
        cls,
        path_base: str,
        do_config: bool = False,
        do_obj: bool = False,
        do_tokens: bool = False,
        do_tokenized: bool = False,
    ) -> "MazeDataset":
        cfg: MazeDatasetConfig | None = None
        if do_config:
            # load config from json
            with open(f"{path_base}/{cls.DISK_SAVE_FILES.cfg}", "r") as f:
                cfg = MazeDatasetConfig.load(json.load(f))

        mazes_objs: list[SolvedMaze] | None = None
        if do_obj:
            raise NotImplementedError("do_obj not implemented")

        mazes_tokens: list[list[str]] | None = None
        if do_tokens:
            # load tokens from jsonl
            with open(f"{path_base}/{cls.DISK_SAVE_FILES.tokens}", "r") as f:
                mazes_tokens = [x.split() for x in f.readlines()]

        loaded_dict: dict | None = None
        if do_tokenized:
            # load tokenized data from npz
            loaded_dict = np.load(
                f"{path_base}/{cls.DISK_SAVE_FILES.tokenized}",
                allow_pickle=False,
            )

            assert "arr" in loaded_dict
            assert "indices" in loaded_dict

        return cls(
            cfg=cfg,
            mazes_objs=mazes_objs,
            mazes_tokens=mazes_tokens,
            mazes_array=None
            if loaded_dict is None
            else IndexedArray(
                arr=torch.tensor(loaded_dict["arr"], device="cpu"),
                indices=torch.tensor(loaded_dict["indices"], device="cpu"),
            ),
        )


MazeDatasetConfig._dataset_class = MazeDataset
