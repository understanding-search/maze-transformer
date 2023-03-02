import pytest

from maze_transformer.evaluation.plot_loss import rolling_average

"""
Test that loading the model and configuration works
* I.e. the outputs of the model are identical when loading into
  a HookedTransformer with folding etc., as they would be from
  just applying the model to the input
"""

def test_rolling_average_smaller_window():
    x = (0, 1, 2, 4, 8)
    assert rolling_average(x, 2) == [0.5, 1.5, 3.0, 6]

