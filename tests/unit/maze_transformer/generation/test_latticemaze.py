import pytest
from numpy.testing import assert_array_equal

from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.generation.latticemaze import LatticeMaze
from maze_transformer.generation.utils import bool_array_from_string
from tests.helpers import utils


def test_as_img():
    connection_list = bool_array_from_string(
        """
        F T T
        T F T
        F F F

        T T F
        T F F
        T F F
        """,
        shape=[2, 3, 3],
    )

    img = LatticeMaze(connection_list=connection_list).as_img()

    expected = bool_array_from_string(
        """
        x x x x x x x
        x _ _ _ _ _ x
        x x x _ x _ x
        x _ _ _ x _ x
        x _ x x x _ x
        x _ _ _ x _ x
        x x x x x x x
        """,
        shape=[7, 7],
        true_symbol="_",
    )

    assert_array_equal(expected, img)


def test_as_adjlist():
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

    adjlist = maze.as_adjlist(shuffle_d0=False, shuffle_d1=False)

    expected = [[[0, 1], [1, 1]], [[0, 0], [0, 1]], [[1, 0], [1, 1]]]

    assert utils.adjlist_to_nested_set(expected) == utils.adjlist_to_nested_set(adjlist)


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
