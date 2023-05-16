from functools import cached_property

import numpy as np

from maze_transformer.dataset.collected_dataset import MazeDatasetCollection, MazeDatasetCollectionConfig, MazeDatasetConfig

DATASET_LENGTHS: list[int] = [1, 0, 3, 2, 1]
DATASET_GRID_SIZES: list[int] = [5, 1, 3, 3, 4]

class TestMazeDatasetCollection():

    @cached_property
    def test_collection(self) -> MazeDatasetCollection:
        config = MazeDatasetCollectionConfig(
            name="test_collection",
            maze_dataset_configs=[
                MazeDatasetConfig(
                    n_mazes=n_mazes,
                    grid_n=grid_n,
                    name=f"test_dataset_{n_mazes}_{grid_n}",
                )
                for n_mazes, grid_n in zip(
                    DATASET_LENGTHS, DATASET_GRID_SIZES
                )
            ]
        )
        return MazeDatasetCollection.from_config(
            config,
            do_generate=True,
            load_local=False,
            do_download=False,
            save_local=True,
            local_base_path="data/",
        )

    def test_dataset_lengths(self):
        assert self.test_collection.dataset_lengths == DATASET_LENGTHS

    def test_dataset_cum_lengths(self):
        assert (self.test_collection.dataset_cum_lengths == np.array([1, 1, 4, 6, 7])).all()

    def test_mazes(self):
        assert len(self.test_collection.mazes) == 7
        assert self.test_collection.mazes[0].connection_list.shape == (2, 5, 5)
        assert self.test_collection.mazes[-1].connection_list.shape == (2, 4, 4)

    def test_len(self):
        assert len(self.test_collection) == 7

    def test_getitem(self):
        assert self.test_collection[0].connection_list.shape == (5, 5)
        assert self.test_collection[1].connection_list.shape == (1, 1)
        assert self.test_collection[6].connection_list.shape == (4, 4)

    def test_download(self):
        # TODO
        pass
         
    def test_serialize_and_load(self):
        serialized = self.test_collection.serialize()
        loaded = MazeDatasetCollection.load(serialized)
        assert loaded.mazes == self.test_collection.mazes
        assert loaded.cfg == self.test_collection.cfg

