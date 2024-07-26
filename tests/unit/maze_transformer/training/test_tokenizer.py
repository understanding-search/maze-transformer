import numpy as np
import pytest
from maze_dataset import SPECIAL_TOKENS, SolvedMaze, utils
from maze_dataset.generation import get_maze_with_solution
from maze_dataset.tokenization import (
    MazeTokenizer,
    MazeTokenizerModular,
    TokenizationMode,
)


@pytest.mark.parametrize(
    "tokenizer",
    [
        pytest.param(tokenizer, id=tokenizer.name)
        for tokenizer in (
            TokenizationMode.AOTP_UT_rasterized.to_legacy_tokenizer(),
            TokenizationMode.AOTP_UT_uniform.to_legacy_tokenizer(),
            MazeTokenizerModular(),
        )
    ],
)
def test_coordinate_system(tokenizer: MazeTokenizer | MazeTokenizerModular):
    """
    Check that the adj_list created by as_tokens() uses the same coordinate system as the LatticeMaze adj_list.

    To test this, generate both adj_lists, sort them and convert to a common format, and check that they are equal.
    """
    maze_size: int = 3
    solved_maze: SolvedMaze = get_maze_with_solution("gen_dfs", (maze_size, maze_size))
    maze_adj_list: np.ndarray = solved_maze.as_adj_list()

    # convert to the same format as the tokenizer adj_list
    maze_adj_list_connections = [
        [f"({c[0]},{c[1]})" for c in conn] for conn in maze_adj_list
    ]

    tokenized_maze: list[str] = solved_maze.as_tokens(tokenizer)

    tokenizer_adj_list = tokenized_maze[
        tokenized_maze.index(SPECIAL_TOKENS.ADJLIST_START)
        + 1 : tokenized_maze.index(SPECIAL_TOKENS.ADJLIST_END)
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
