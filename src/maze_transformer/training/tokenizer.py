from dataclasses import dataclass, field
from itertools import chain

from maze_transformer.generation.latticemaze import (SPECIAL_TOKENS,
                                                     CoordArray, CoordTup,
                                                     LatticeMaze)

# pylint: disable=unused-import


@dataclass(frozen=True, kw_only=True)
class MazeTokenizer:
    """solved maze for serialization"""

    maze: LatticeMaze
    solution: CoordArray
    metadata: dict = field(default_factory=dict)

    pos_start = property(lambda self: self.solution[0])
    pos_end = property(lambda self: self.solution[-1])

    def as_tokens(
        self,
        node_token_map: dict[CoordTup, str],
        solution: bool = True,
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
                    for c_s, c_e in self.maze.as_adjlist()
                ]
            ),
            SPECIAL_TOKENS["adjlist_end"],
            # give target
            SPECIAL_TOKENS["target_start"],
            node_token_map[tuple(self.pos_end)],
            SPECIAL_TOKENS["target_end"],
        ]

        if solution:
            # give path
            tokens.extend(
                [
                    SPECIAL_TOKENS["start_path"],
                    *[node_token_map[tuple(c.tolist())] for c in self.solution],
                    SPECIAL_TOKENS["end_path"],
                ]
            )

        return tokens
