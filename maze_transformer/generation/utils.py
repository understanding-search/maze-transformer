import math
from typing import List, Tuple

import numpy as np
from muutils.tensor_utils import NDArray

from generation.latticemaze import Coord, LatticeMaze


def bool_array_from_string(
    string: str, shape: list[int], true_symbol: str = "T"
) -> NDArray:
    """Transform a string into an ndarray of bools.

    Parameters
    ----------
    string: str
        The string representation of the array
    shape: list[int]
        The shape of the resulting array
    true_symbol:
        The character to parse as True. Whitespace will be removed. All other characters will be parsed as False.

    Returns
    -------
    NDArray
        A ndarray with dtype bool.

    Examples
    --------
    >>> bool_array_from_string(
    ...     "TT TF", shape=[2,2]
    ... )
    array([[ True,  True],
        [ True, False]])
    """
    stripped = "".join(string.split())

    expected_symbol_count = math.prod(shape)
    symbol_count = len(stripped)
    if len(stripped) != expected_symbol_count:
        raise ValueError(
            f"Connection List contains the wrong number of symbols. Expected {expected_symbol_count}. Found {symbol_count} in {stripped}."
        )

    bools = [True if symbol == true_symbol else False for symbol in stripped]
    return np.array(bools).reshape(*shape)


def inside(p: Coord, width: int, height: int) -> bool:
    x, y = p
    return 0 <= x < width and 0 <= y < height


def neighbor(current: Coord, direction: int, maze: LatticeMaze) -> Coord:
    x, y = current

    if direction == 0:
        x -= 1  # Left
    elif direction == 1:
        x += 1  # Right
    elif direction == 2:
        y -= 1  # Up
    elif direction == 3:
        y += 1  # Down
    else:
        return None

    # neighbor_x, neighbor_y = maze.connection_list[:][x][y]

    return maze.connection_list[:][x][y] if 0 <= x < maze.length and 0 <= y < maze[x].length else None

