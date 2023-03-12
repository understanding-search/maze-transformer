import numpy as np

from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.training.mazedataset import MazeDatasetConfig
from maze_transformer.training.tokenizer import SPECIAL_TOKENS, MazeTokenizer


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

    node_token_map = MazeDatasetConfig(
        grid_n=maze_size, name="test", n_mazes=1
    ).node_token_map

    tokenized_maze = MazeTokenizer(
        maze=maze,
        solution=np.array(
            maze.find_shortest_path(
                c_start=(0, 0),
                c_end=(maze_size - 1, maze_size - 1),
            )
        ),
    ).as_tokens(node_token_map)

    tokenizer_adjlist = tokenized_maze[
        1 : tokenized_maze.index(SPECIAL_TOKENS["adjlist_end"])
    ]

    # There are 4 tokens per connection, only the first and third are coords
    tokenizer_adjlist_connections = [
        [tokenizer_adjlist[i], tokenizer_adjlist[i + 2]]
        for i in range(0, len(tokenizer_adjlist), 4)
    ]

    # sort both adjlists
    maze_adjlist_connections = sorted(
        [sorted(conn) for conn in maze_adjlist_connections]
    )
    tokenizer_adjlist_connections = sorted(
        [sorted(conn) for conn in tokenizer_adjlist_connections]
    )

    assert tokenizer_adjlist_connections == maze_adjlist_connections
