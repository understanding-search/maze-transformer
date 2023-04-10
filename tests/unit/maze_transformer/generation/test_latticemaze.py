import pytest

from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.generation.lattice_maze import LatticeMaze
from maze_transformer.generation.utils import bool_array_from_string
from tests.helpers import utils


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
    maze = LatticeMazeGenerators.gen_dfs((3, 2))
    assert maze.get_nodes() == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]


def test_generate_random_path():
    maze = LatticeMazeGenerators.gen_dfs((2, 2))
    path = maze.generate_random_path()

    # len > 1 ensures that we have unique start and end nodes
    assert len(path) > 1


def test_generate_random_path_size_1():
    maze = LatticeMazeGenerators.gen_dfs((1, 1))
    with pytest.raises(AssertionError):
        maze.generate_random_path()
