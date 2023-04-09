import numpy as np
import pytest

from maze_transformer.generation.generators import LatticeMazeGenerators, GENERATORS_MAP
from maze_transformer.generation.lattice_maze import LatticeMaze
from maze_transformer.generation.utils import bool_array_from_string
from tests.helpers import utils


def test_pixels_ascii_roundtrip():
    """tests all generators work and can be written to/from ascii and pixels"""
    n: int = 5
    for maze_gen_func in GENERATORS_MAP.values():
        maze: LatticeMaze = maze_gen_func(np.array([n,n]))

        maze_pixels: np.ndarray = maze.as_pixels()
        maze_ascii: str = maze.as_ascii()

        assert maze == LatticeMaze.from_pixels(maze_pixels)
        assert maze == LatticeMaze.from_ascii(maze_ascii)

        assert maze_pixels.shape == (n*2+1, n*2+1), maze_pixels.shape
        assert all(n*2+1 == len(line) for line in maze_ascii.splitlines()), maze_ascii


def test_as_adj_list():
    connection_list = bool_array_from_string(
        """
        F T
        F F

        T F
        T F
        """,
        shape=[2, 2, 2],
    )

    maze = LatticeMaze(connection_list=connection_list)

    adj_list = maze.as_adj_list(shuffle_d0=False, shuffle_d1=False)

    expected = [[[0, 1], [1, 1]], [[0, 0], [0, 1]], [[1, 0], [1, 1]]]

    assert utils.adj_list_to_nested_set(expected) == utils.adj_list_to_nested_set(
        adj_list
    )


def test_get_nodes():
    for maze_gen_func in GENERATORS_MAP.values():
        maze = LatticeMazeGenerators.gen_dfs((3, 2))
        assert maze.get_nodes() == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]


def test_generate_random_path():
    for maze_gen_func in GENERATORS_MAP.values():
        maze = LatticeMazeGenerators.gen_dfs((2, 2))
        path = maze.generate_random_path()

        # len > 1 ensures that we have unique start and end nodes
        assert len(path) > 1


def test_generate_random_path_size_1():
    for maze_gen_func in GENERATORS_MAP.values():
        maze = LatticeMazeGenerators.gen_dfs((1, 1))
        with pytest.raises(AssertionError):
            maze.generate_random_path()
