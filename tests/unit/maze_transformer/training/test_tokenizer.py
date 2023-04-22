from maze_transformer.generation.constants import SPECIAL_TOKENS
from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.generation.lattice_maze import SolvedMaze
from maze_transformer.training.maze_dataset import MazeDatasetConfig
from tests.helpers import utils


def test_coordinate_system():
    """
    Check that the adj_list created by to_tokens() uses the same coordinate system as the LatticeMaze adj_list.

    To test this, generate both adj_lists, sort them and convert to a common format, and check that they are equal.
    """
    maze_size = 3
    solved_maze: SolvedMaze = LatticeMazeGenerators.gen_dfs_with_solution(
        (maze_size, maze_size)
    )
    maze_adj_list = solved_maze.as_adj_list()

    # convert to the same format as the tokenizer adj_list
    maze_adj_list_connections = [
        [f"({c[0]},{c[1]})" for c in conn] for conn in maze_adj_list
    ]

    # See https://github.com/AISC-understanding-search/maze-transformer/issues/77
    node_token_map = MazeDatasetConfig(
        grid_n=maze_size, name="test", n_mazes=1
    ).node_token_map

    tokenized_maze = solved_maze.to_tokens(node_token_map)

    tokenizer_adj_list = tokenized_maze[
        tokenized_maze.index(SPECIAL_TOKENS["adj_list_start"])
        + 1 : tokenized_maze.index(SPECIAL_TOKENS["adj_list_end"])
    ]

    # remove special tokens
    tokenizer_adj_list_coordinates = [
        token for token in tokenizer_adj_list if token not in SPECIAL_TOKENS.values()
    ]

    # Group pairs of coordinates
    tokenizer_adj_list_connections = [
        *zip(tokenizer_adj_list_coordinates[::2], tokenizer_adj_list_coordinates[1::2])
    ]

    assert utils.adj_list_to_nested_set(
        tokenizer_adj_list_connections
    ) == utils.adj_list_to_nested_set(maze_adj_list_connections)
