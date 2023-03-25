import typing

from maze_transformer.generation.constants import SPECIAL_TOKENS


def tokens_between(tokens: list[str], start_value: str, end_value: str) -> list[str]:
    start_idx = tokens.index(start_value) + 1
    end_idx = tokens.index(end_value)

    return tokens[start_idx:end_idx]


def get_adjlist_tokens(tokens: list[str]) -> list[str]:
    return tokens_between(
        tokens, SPECIAL_TOKENS["adjlist_start"], SPECIAL_TOKENS["adjlist_end"]
    )


def get_path_tokens(tokens: list[str]) -> list[str]:
    """The path is considered everything from the first path coord to the end of the list, including the path_end token (ie everything we are asking the model to predict)"""
    start_idx = tokens.index(SPECIAL_TOKENS["path_start"]) + 1
    return tokens[start_idx:]


def get_origin_token(tokens: list[str]) -> str:
    return tokens_between(
        tokens, SPECIAL_TOKENS["origin_start"], SPECIAL_TOKENS["origin_end"]
    )[0]


def get_target_token(tokens: list[str]) -> str:
    return tokens_between(
        tokens, SPECIAL_TOKENS["target_start"], SPECIAL_TOKENS["target_end"]
    )[0]


def get_tokens_up_to_path_start(
    tokens: list[str], include_start_coord: bool = True
) -> list[str]:
    path_start_idx: int = tokens.index(SPECIAL_TOKENS["path_start"]) + 1
    if include_start_coord:
        return tokens[: path_start_idx + 1]
    else:
        return tokens[:path_start_idx]


def decode_maze_tokens_to_coords(
    tokens: list[str],
    mazedata_cfg,  # TODO: cannot type this right now because importing MazeDatasetConfig causes a circular import
    when_noncoord: typing.Literal["except", "skip", "include"] = "skip",
) -> list[str | tuple[int, int]]:
    """given a list of tokens, decode the coordinate-tokens to a list of coordinates, leaving other tokens as-is"""
    output: list[str | tuple[int, int]] = list()
    for idx, tk in enumerate(tokens):
        if tk in mazedata_cfg.token_node_map:
            output.append(mazedata_cfg.token_node_map[tk])
        else:
            if when_noncoord == "skip":
                continue
            elif when_noncoord == "include":
                output.append(tk)
            elif when_noncoord == "except":
                raise ValueError(f"token '{tk}' at {idx = } is not a coordinate")
            else:
                raise ValueError(f"invalid value for {when_noncoord = }")
    return output
