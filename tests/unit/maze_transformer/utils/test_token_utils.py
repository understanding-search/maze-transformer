import pytest

import maze_transformer.utils.token_utils as token_utils
from maze_transformer.training.config import MazeDatasetConfig

MAZE_TOKENS = "<ADJLIST_START> (0,1) <--> (1,1) ; (1,0) <--> (1,1) ; (0,1) <--> (0,0) ; <ADJLIST_END> <ORIGIN_START> (1,0) <ORIGIN_END> <TARGET_START> (1,1) <TARGET_END> <PATH_START> (1,0) (1,1) <PATH_END>".split()


def test_tokens_between():
    result = token_utils.tokens_between(MAZE_TOKENS, "<PATH_START>", "<PATH_END>")
    assert result == ["(1,0)", "(1,1)"]


def test_tokens_between_out_of_order():
    with pytest.raises(AssertionError):
        token_utils.tokens_between(MAZE_TOKENS, "<PATH_END>", "<PATH_START>")


def test_get_adjlist_tokens():
    result = token_utils.get_adjlist_tokens(MAZE_TOKENS)
    expected = "(0,1) <--> (1,1) ; (1,0) <--> (1,1) ; (0,1) <--> (0,0) ;".split()
    assert result == expected


def test_get_path_tokens():
    result = token_utils.get_path_tokens(MAZE_TOKENS)
    assert result == ["(1,0)", "(1,1)", "<PATH_END>"]


def test_get_origin_token():
    result = token_utils.get_origin_token(MAZE_TOKENS)
    assert result == "(1,0)"


def test_get_target_token():
    result = token_utils.get_target_token(MAZE_TOKENS)
    assert result == "(1,1)"


def test_get_tokens_up_to_path_start_including_start():
    result = token_utils.get_tokens_up_to_path_start(MAZE_TOKENS)
    expected = "<ADJLIST_START> (0,1) <--> (1,1) ; (1,0) <--> (1,1) ; (0,1) <--> (0,0) ; <ADJLIST_END> <ORIGIN_START> (1,0) <ORIGIN_END> <TARGET_START> (1,1) <TARGET_END> <PATH_START> (1,0)".split()
    assert result == expected


def test_get_tokens_up_to_path_start_excluding_start():
    result = token_utils.get_tokens_up_to_path_start(
        MAZE_TOKENS, include_start_coord=False
    )
    expected = "<ADJLIST_START> (0,1) <--> (1,1) ; (1,0) <--> (1,1) ; (0,1) <--> (0,0) ; <ADJLIST_END> <ORIGIN_START> (1,0) <ORIGIN_END> <TARGET_START> (1,1) <TARGET_END> <PATH_START>".split()
    assert result == expected


def test_decode_maze_tokens_to_coords():
    adj_list = token_utils.get_adjlist_tokens(MAZE_TOKENS)
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