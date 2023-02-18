import pytest

from maze_transformer.evaluation.plot_loss import rolling_average


def test_rolling_average_smaller_window():
    x = (0, 1, 2, 4, 8)
    assert rolling_average(x, 2) == [0.5, 1.5, 3.0, 6]


def test_rolling_average_same_size():
    x = (1, 2, 3, 4, 5)
    assert rolling_average(x, 5) == [3.0]


def test_rolling_average_larger_window():
    with pytest.raises(ValueError):
        rolling_average((1, 2, 3), 5)
