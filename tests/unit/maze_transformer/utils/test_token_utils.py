import pytest

import maze_transformer.utils.token_utils as token_utils
from maze_transformer.training.config import MazeDatasetConfig

maze_tokens = [
    "<ADJLIST_START>",
    "(0,1)",
    "<-->",
    "(1,1)",
    ";",
    "(1,0)",
    "<-->",
    "(1,1)",
    ";",
    "(0,1)",
    "<-->",
    "(0,0)",
    ";",
    "<ADJLIST_END>",
    "<ORIGIN_START>",
    "(1,0)",
    "<ORIGIN_END>",
    "<TARGET_START>",
    "(1,1)",
    "<TARGET_END>",
    "<PATH_START>",
    "(1,0)",
    "(1,1)",
    "<PATH_END>",
]


def test_tokens_between():
    result = token_utils.tokens_between(maze_tokens, "<PATH_START>", "<PATH_END>")
    assert result == ["(1,0)", "(1,1)"]


def test_get_adjlist_tokens():
    result = token_utils.get_adjlist_tokens(maze_tokens)
    assert result == [
        "(0,1)",
        "<-->",
        "(1,1)",
        ";",
        "(1,0)",
        "<-->",
        "(1,1)",
        ";",
        "(0,1)",
        "<-->",
        "(0,0)",
        ";",
    ]


def test_get_path_tokens():
    result = token_utils.get_path_tokens(maze_tokens)
    assert result == ["(1,0)", "(1,1)", "<PATH_END>"]


def test_get_origin_token():
    result = token_utils.get_origin_token(maze_tokens)
    assert result == "(1,0)"


def test_get_target_token():
    result = token_utils.get_target_token(maze_tokens)
    assert result == "(1,1)"


def test_get_tokens_up_to_path_start_including_start():
    result = token_utils.get_tokens_up_to_path_start(maze_tokens)
    assert result == [
        "<ADJLIST_START>",
        "(0,1)",
        "<-->",
        "(1,1)",
        ";",
        "(1,0)",
        "<-->",
        "(1,1)",
        ";",
        "(0,1)",
        "<-->",
        "(0,0)",
        ";",
        "<ADJLIST_END>",
        "<ORIGIN_START>",
        "(1,0)",
        "<ORIGIN_END>",
        "<TARGET_START>",
        "(1,1)",
        "<TARGET_END>",
        "<PATH_START>",
        "(1,0)",
    ]


def test_get_tokens_up_to_path_start_excluding_start():
    result = token_utils.get_tokens_up_to_path_start(
        maze_tokens, include_start_coord=False
    )
    assert result == [
        "<ADJLIST_START>",
        "(0,1)",
        "<-->",
        "(1,1)",
        ";",
        "(1,0)",
        "<-->",
        "(1,1)",
        ";",
        "(0,1)",
        "<-->",
        "(0,0)",
        ";",
        "<ADJLIST_END>",
        "<ORIGIN_START>",
        "(1,0)",
        "<ORIGIN_END>",
        "<TARGET_START>",
        "(1,1)",
        "<TARGET_END>",
        "<PATH_START>",
    ]


def test_decode_maze_tokens_to_coords():
    adj_list = token_utils.get_adjlist_tokens(maze_tokens)
    config = MazeDatasetConfig(name="test", grid_n=2, n_mazes=1)

    skipped = token_utils.decode_maze_tokens_to_coords(
        adj_list, config, when_noncoord="skip"
    )

    included = token_utils.decode_maze_tokens_to_coords(
        adj_list, config, when_noncoord="include"
    )

    assert skipped == [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, 1),
        (0, 1),
        (0, 0),
    ]

    assert included == [
        (0, 1),
        "<-->",
        (1, 1),
        ";",
        (1, 0),
        "<-->",
        (1, 1),
        ";",
        (0, 1),
        "<-->",
        (0, 0),
        ";",
    ]

    with pytest.raises(ValueError, match="not a coordinate"):
        token_utils.decode_maze_tokens_to_coords(
            adj_list, config, when_noncoord="except"
        )
