import numpy as np
import pytest

from maze_transformer.generation.constants import CoordArray
from maze_transformer.generation.lattice_maze import SolvedMaze
from maze_transformer.generation.utils import bool_array_from_string
from maze_transformer.training.dataset import (
    SaveFormats,
    register_filter_namespace_for_dataset,
)
from maze_transformer.training.maze_dataset import (
    MazeDataset,
    MazeDatasetConfig,
    register_wrap_solved_maze_filter,
)


class TestMazeDatasetConfig:
    pass


class TestMazeDataset:
    config = MazeDatasetConfig(name="test", grid_n=3, n_mazes=5)

    def test_generate_serial(self):
        dataset = MazeDataset.generate(self.config, gen_parallel=False)

        assert len(dataset) == 5
        for i, maze in enumerate(dataset):
            maze_obj = SolvedMaze.from_tokens(maze.split(" "), self.config)
            assert maze_obj.grid_shape == (3, 3)
            assert (
                maze_obj
                == SolvedMaze.from_tokens(dataset.mazes_tokens[i], self.config)
                == dataset.mazes_objs[i]
            )

    def test_generate_parallel(self):
        dataset = MazeDataset.generate(self.config, gen_parallel=True, verbose=True)

        assert len(dataset) == 5
        for i, maze in enumerate(dataset):
            maze_obj = SolvedMaze.from_tokens(maze.split(" "), self.config)
            assert maze_obj.grid_shape == (3, 3)
            assert (
                maze_obj
                == SolvedMaze.from_tokens(dataset.mazes_tokens[i], self.config)
                == dataset.mazes_objs[i]
            )

    def test_data_hash(self):
        dataset = MazeDataset.generate(self.config)
        # TODO: dataset.data_hash doesn't work right now

    def test_get(self):
        dataset = MazeDataset.generate(self.config)

        assert dataset.get(2) == dataset.mazes[2]
        # TODO: tokens are not equal due to `get` not caching and shuffling
        # assert dataset.get(2, SaveFormats.TOKENS) == dataset.mazes_tokens[2]

        with pytest.raises(NotImplementedError):
            dataset.get(2, SaveFormats.ARRAY)

        with pytest.raises(ValueError):
            dataset.get(2, "foo")  # type: ignore

    def test_download(self):
        with pytest.raises(NotImplementedError):
            MazeDataset.download(self.config)

    def test_serialize_load(self):
        dataset = MazeDataset.generate(self.config)
        dataset_copy = MazeDataset.load(dataset.serialize())

        assert dataset.cfg == dataset_copy.cfg
        # TODO: This is currently failing due to dataset yielding shuffled token strings. Need to decide if this is desired behaviour
        # for maze, maze_copy in zip(dataset, dataset_copy):
        #     assert maze == maze_copy

    # TODO: do this after testing default filters
    def test_custom_maze_filter(self):
        connection_list = bool_array_from_string(
            """
            F T
            F F

            T F
            T F
            """,
            shape=[2, 2, 2],
        )
        solutions = [
            [[0, 0], [0, 1], [1, 1]],
            [[0, 0], [0, 1]],
            [[0, 0]],
        ]

        mazes = [SolvedMaze(connection_list, solution) for solution in solutions]
        dataset = MazeDataset(self.config, mazes)
        filtered = dataset.custom_maze_filter(lambda m: len(m.solution) == 1)
        assert filtered.mazes_objs == [mazes[2]]


class TestMazeDatasetFilters:
    config = MazeDatasetConfig(name="test", grid_n=3, n_mazes=5)
    connection_list = bool_array_from_string(
        """
        F T
        F F

        T F
        T F
        """,
        shape=[2, 2, 2],
    )

    def test_register_wrap_solved_maze_filter(self):
        class TestDataset(MazeDataset):
            ...

        @register_filter_namespace_for_dataset(TestDataset)
        class TestFilters:
            @register_wrap_solved_maze_filter
            @staticmethod
            def solution_match(maze: SolvedMaze, solution: CoordArray) -> bool:
                """Test for solution equality"""
                return (maze.solution == solution).all()

        maze1 = SolvedMaze(
            connection_list=self.connection_list, solution=np.array([[0, 0]])
        )
        maze2 = SolvedMaze(
            connection_list=self.connection_list, solution=np.array([[0, 1]])
        )

        dataset = TestDataset(self.config, [maze1, maze2])
        filtered = dataset.filter_by.solution_match(solution=np.array([[0, 0]]))
        assert filtered.mazes_objs == [maze1]

    def test_path_length(self):
        long_maze = SolvedMaze(
            connection_list=self.connection_list,
            solution=np.array([[0, 0], [0, 1], [1, 1]]),
        )

        short_maze = SolvedMaze(
            connection_list=self.connection_list, solution=np.array([[0, 0], [0, 1]])
        )

        dataset = MazeDataset(self.config, [long_maze, short_maze])
        path_length_filtered = dataset.filter_by.path_length(min_length=3)
        start_end_filtered = dataset.filter_by.start_end_distance(min_distance=2)

        assert type(path_length_filtered) == type(dataset)
        assert path_length_filtered.mazes_objs == [long_maze]
        assert start_end_filtered.mazes_objs == [long_maze]
        assert dataset.mazes_objs == [long_maze, short_maze]

    def test_cut_percentile_shortest(self):
        solutions = [
            [[0, 0], [0, 1], [1, 1]],
            [[0, 0], [0, 1]],
            [[0, 0]],
        ]

        mazes = [SolvedMaze(self.connection_list, solution) for solution in solutions]
        dataset = MazeDataset(self.config, mazes)
        # TODO: this throws an exception
        filtered = dataset.filter_by.cut_percentile_shortest(percentile=0.49)

        assert filtered.mazes == mazes[:2]
