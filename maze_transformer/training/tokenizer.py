from itertools import chain

from maze_transformer.generation.constants import SPECIAL_TOKENS, Coord, CoordTup
from maze_transformer.generation.latticemaze import LatticeMaze

# pylint: disable=unused-import


def maze_to_tokens(
    maze: LatticeMaze,
    solution: list[Coord],
    node_token_map: dict[CoordTup, str],
) -> list[str]:
    """serialize maze and solution to tokens"""
    tokens: list[str] = [
        # give adjacency list
        SPECIAL_TOKENS["adjlist_start"],
        *chain.from_iterable(
            [
                [
                    node_token_map[tuple(c_s.tolist())],
                    SPECIAL_TOKENS["connector"],
                    node_token_map[tuple(c_e.tolist())],
                    SPECIAL_TOKENS["adjacency_endline"],
                ]
                for c_s, c_e in maze.as_adj_list()
            ]
        ),
        SPECIAL_TOKENS["adjlist_end"],
        # give origin
        SPECIAL_TOKENS["origin_start"],
        node_token_map[tuple(solution[0])],
        SPECIAL_TOKENS["origin_end"],
        # give target
        SPECIAL_TOKENS["target_start"],
        node_token_map[tuple(solution[-1])],
        SPECIAL_TOKENS["target_end"],
        SPECIAL_TOKENS["path_start"],
        *[node_token_map[tuple(c.tolist())] for c in solution],
        SPECIAL_TOKENS["path_end"],
    ]

    return tokens
