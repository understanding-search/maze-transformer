import numpy as np
from maze_dataset import LatticeMaze
from maze_dataset.utils import bool_array_from_string

from maze_transformer.evaluation.path_evals import PathEvals


def test_node_overlap_short_match():
    # this will happen if origin and target are the same. Correct behaviour would be an immediate path_end
    prediction = np.array([(1, 2)])
    solution = np.array([(1, 2)])

    assert PathEvals.node_overlap(solution, prediction) == 1.0


def test_node_overlap():
    prediction = np.array([(1, 2), (2, 3), (1, 1)])
    solution = np.array([(2, 2), (1, 2)])

    assert PathEvals.node_overlap(solution, prediction) == 0.5


def test_num_connections_adjacent_lattice():
    prediction = np.array([(0, 0), (0, 1), (1, 2)])

    assert PathEvals.num_connections_adjacent_lattice(prediction) == 1.0


def fraction_connections_adjacent_lattice():
    prediction = np.array([(0, 0), (0, 1), (1, 2)])

    assert PathEvals.num_connections_adjacent_lattice(prediction) == 0.5


def test_num_connections_adjacent():
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
    # first pair is not connected, 2nd pair is
    prediction = np.array([(0, 0), (1, 0), (1, 1)])

    assert PathEvals.num_connections_adjacent(maze, prediction) == 1.0


def test_fraction_connections_adjacent():
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
    # first pair is not connected, 2nd pair is
    prediction = np.array([(0, 0), (1, 0), (1, 1)])
    short_prediction = np.array([(0, 0)])

    assert PathEvals.fraction_connections_adjacent(maze, prediction) == 0.5
    assert PathEvals.fraction_connections_adjacent(maze, short_prediction) == 0.0


def test_exact_path_predicted():
    solution = np.array([(0, 0), (0, 1), (1, 1)])
    good_prediction = solution
    bad_prediction = np.array([(0, 0), (1, 1)])

    assert PathEvals.exact_path_predicted(solution, good_prediction) == 1.0
    assert PathEvals.exact_path_predicted(solution, bad_prediction) == 0.0


def test_solution_length():
    solution = np.array([(0, 0), (0, 1)])

    assert PathEvals.solution_length(solution) == 2


def test_streak_length_until_incorrect():
    solution = np.array([(0, 0), (0, 1), (0, 2)])
    terrible_prediction = np.array([(0, 1)])
    bad_prediction = np.array([(0, 0), (0, 1), (2, 2)])
    long_prediction = np.concatenate([solution, np.array([(0, 2), (0, 2)])])

    assert PathEvals.streak_length_until_incorrect(solution, terrible_prediction) == 0.0
    assert PathEvals.streak_length_until_incorrect(solution, bad_prediction) == 2.0
    assert PathEvals.streak_length_until_incorrect(solution, solution) == 3.0
    assert PathEvals.streak_length_until_incorrect(solution, long_prediction) == 3.0
