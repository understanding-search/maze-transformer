from typing import Any, Iterable, Literal

from maze_transformer.generation.constants import SPECIAL_TOKENS, CoordTup

WhenMissing = Literal["except", "skip", "include"]


def tokens_between(tokens: list[str], start_value: str, end_value: str) -> list[str]:
    start_idx = tokens.index(start_value) + 1
    end_idx = tokens.index(end_value)

    assert start_idx < end_idx, "Start must come before end"

    return tokens[start_idx:end_idx]


def get_adj_list_tokens(tokens: list[str]) -> list[str]:
    return tokens_between(
        tokens, SPECIAL_TOKENS["adj_list_start"], SPECIAL_TOKENS["adj_list_end"]
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


def apply_mapping(
    iter: Iterable[Any],
    mapping: dict[Any, Any],
    when_missing: WhenMissing = "skip",
) -> list[Any]:
    """Given a list and a mapping, apply the mapping to the list"""
    output = list()
    for item in iter:
        if item in mapping:
            output.append(mapping[item])
            continue
        match when_missing:
            case "skip":
                continue
            case "include":
                output.append(item)
            case "except":
                raise ValueError(f"item {item} is missing from mapping {mapping}")
            case _:
                raise ValueError(f"invalid value for {when_missing = }")
    return output


def tokens_to_coords(
    tokens: list[str],
    maze_data_cfg,  # TODO: cannot type this right now because importing MazeDatasetConfig causes a circular import
    when_noncoord: WhenMissing = "skip",
) -> list[str | CoordTup]:
    return apply_mapping(tokens, maze_data_cfg.token_node_map, when_noncoord)


def coords_to_tokens(
    coords: list[str | CoordTup],
    maze_data_cfg,  # TODO: cannot type this right now because importing MazeDatasetConfig causes a circular import
    when_noncoord: WhenMissing = "skip",
) -> list[str]:
    return apply_mapping(coords, maze_data_cfg.node_token_map, when_noncoord)
