import numpy as np

from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.training.mazedataset import MazeDatasetConfig
from maze_transformer.training.tokenizer import SPECIAL_TOKENS, MazeTokenizer
from tests.helpers import utils


def test_coordinate_system():
    """
    Check that the adjlist created by .as_tokens() uses the same coordinate system as the LatticeMaze adjlist.

    To test this, generate both adjlists, sort them and convert to a common format, and check that they are equal.
    """
    maze_size = 3
    maze = LatticeMazeGenerators.gen_dfs((maze_size, maze_size))
    maze_adjlist = maze.as_adjlist()

    # convert to the same format as the tokenizer adjlist
    maze_adjlist_connections = [
        [f"({c[0]},{c[1]})" for c in conn] for conn in maze_adjlist
    ]

    # See https://github.com/AISC-understanding-search/maze-transformer/issues/77
    node_token_map = MazeDatasetConfig(
        grid_n=maze_size, name="test", n_mazes=1
    ).node_token_map

    tokenized_maze = MazeTokenizer(
        maze=maze,
        solution=np.array([[0, 0]]),
    ).as_tokens(node_token_map)

    tokenizer_adjlist = tokenized_maze[
        tokenized_maze.index(SPECIAL_TOKENS["adjlist_start"])
        + 1 : tokenized_maze.index(SPECIAL_TOKENS["adjlist_end"])
    ]

    # remove special tokens
    tokenizer_adjlist_coordinates = [
        token for token in tokenizer_adjlist if token not in SPECIAL_TOKENS.values()
    ]

    # Group pairs of coordinates
    tokenizer_adjlist_connections = [
        *zip(tokenizer_adjlist_coordinates[::2], tokenizer_adjlist_coordinates[1::2])
    ]

    assert utils.adjlist_to_nested_set(
        tokenizer_adjlist_connections
    ) == utils.adjlist_to_nested_set(maze_adjlist_connections)
