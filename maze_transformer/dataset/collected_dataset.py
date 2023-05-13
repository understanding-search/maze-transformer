import itertools
from functools import cached_property

import numpy as np
from muutils.json_serialize import JSONitem, serializable_dataclass, serializable_field

from maze_transformer.dataset.dataset import GPTDataset, GPTDatasetConfig
from maze_transformer.dataset.maze_dataset import (
    _MAZEDATASET_PROPERTIES_TO_SERIALIZE,
    MazeDataset,
    MazeDatasetConfig,
)
from maze_transformer.generation.lattice_maze import LatticeMaze


@serializable_dataclass(
    kw_only=True, properties_to_serialize=_MAZEDATASET_PROPERTIES_TO_SERIALIZE
)
class MazeDatasetCollectionConfig(GPTDatasetConfig):
    """maze dataset collection configuration, including tokenizers and shuffle"""

    maze_dataset_configs: list[MazeDatasetConfig] = serializable_field(
        serialization_fn=lambda configs: [config.serialize() for config in configs],
        loading_fn=lambda data: [
            MazeDatasetConfig.load(config) for config in data["maze_dataset_configs"]
        ],
    )

    @property
    def n_mazes(self) -> int:
        return sum(config.n_mazes for config in self.maze_dataset_configs)

    @property
    def max_grid_n(self) -> int:
        return max(config.max_grid_n for config in self.maze_dataset_configs)


class MazeDatasetCollection(GPTDataset):
    """a collection of maze datasets"""

    def __init__(
        self,
        cfg: MazeDatasetCollectionConfig,
        maze_datasets: list[MazeDataset],
    ) -> None:
        super().__init__()
        self.cfg: MazeDatasetCollectionConfig = cfg
        self.maze_datasets: list[MazeDataset] = list(maze_datasets)

    @cached_property
    def dataset_lengths(self) -> list[int]:
        return [len(dataset) for dataset in self.maze_datasets]

    @cached_property
    def dataset_cum_lengths(self) -> list[int]:
        return list(itertools.accumulate(self.dataset_lengths))

    @cached_property
    def mazes(self) -> list[LatticeMaze]:
        return list(
            itertools.chain.from_iterable(
                dataset.mazes for dataset in self.maze_datasets
            )
        )

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.maze_datasets)

    def __getitem__(self, index: int):
        # find which dataset the index belongs to
        dataset_idx: int = np.searchsorted(self.dataset_cum_lengths, index)
        return self.maze_datasets[dataset_idx][
            index - self.dataset_cum_lengths[dataset_idx - 1]
        ]

    @classmethod
    def generate(
        cls, cfg: MazeDatasetCollectionConfig, **kwargs
    ) -> "MazeDatasetCollection":
        datasets = [
            MazeDataset.generate(config, **kwargs)
            for config in cfg.maze_dataset_configs
        ]
        return cls(cfg, datasets)

    @classmethod
    def download(
        cls, cfg: MazeDatasetCollectionConfig, **kwargs
    ) -> "MazeDatasetCollection":
        datasets = [
            MazeDataset.download(config, **kwargs)
            for config in cfg.maze_dataset_configs
        ]
        return cls(cfg, datasets)

    def serialize(self) -> JSONitem:
        return dict(
            cfg=self.cfg.serialize(),
            maze_datasets=[dataset.serialize() for dataset in self.maze_datasets],
        )

    @classmethod
    def load(cls, data: JSONitem) -> "MazeDatasetCollection":
        cfg = MazeDatasetCollectionConfig.from_dict(data["cfg"])
        datasets = [
            MazeDataset.load(dataset_data) for dataset_data in data["maze_datasets"]
        ]
        return cls(cfg, datasets)

    def update_self_config(self) -> None:
        self.cfg.n_mazes = len(self)
        for dataset in self.maze_datasets:
            dataset.update_self_config()
