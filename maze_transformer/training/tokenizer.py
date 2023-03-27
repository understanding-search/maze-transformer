from dataclasses import dataclass, field
from itertools import chain

# Avoid circular import from training/config.py
from typing import TYPE_CHECKING, Union  # need Union as "a" | "b" doesn't work

import torch
from muutils.tensor_utils import ATensor, NDArray
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import BatchEncoding

from maze_transformer.generation.latticemaze import (
    SPECIAL_TOKENS,
    CoordArray,
    CoordTup,
    LatticeMaze,
)

if TYPE_CHECKING:
    from maze_transformer.training.config import ConfigHolder, MazeDatasetConfig
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
            # give origin
            SPECIAL_TOKENS["origin_start"],
            node_token_map[tuple(self.pos_start)],
            SPECIAL_TOKENS["origin_end"],
            # give target
            SPECIAL_TOKENS["target_start"],
            node_token_map[tuple(self.pos_end)],
            SPECIAL_TOKENS["target_end"],
        ]

        if solution:
            # give path
            tokens.extend(
                [
                    SPECIAL_TOKENS["path_start"],
                    *[node_token_map[tuple(c.tolist())] for c in self.solution],
                    SPECIAL_TOKENS["path_end"],
                ]
            )

        return tokens


class HuggingMazeTokenizer(PreTrainedTokenizer):
    vocab: dict[str, int]  # map of token_ids to strings

    bos_token: str = SPECIAL_TOKENS["path_start"]
    eos_token: str = SPECIAL_TOKENS["path_end"]
    pad_token: str = SPECIAL_TOKENS["padding"]
    unk_token: str = "<UNK>"

    vocab_size: int = 0
    additional_special_tokens: list[str] = [
        x for x in SPECIAL_TOKENS.values() if x not in [SPECIAL_TOKENS["padding"]]
    ]

    # Overwrite class attributes
    padding_side = "left"
    truncation_side = "left"  #! strange choice, but it's what we did in pad_sequence

    name_or_path = "maze_tokenizer"

    def __init__(
        self, 
        cfg: Union["ConfigHolder", "MazeDatasetConfig", None] = None, 
        token_arr: list[str]|None = None,
        seq_len_max: int|None = None,
        **kwargs,
    ) -> None:
        """takes either a cfg, or a token_arr and seq_len_max. also, kwargs are passed to super `PreTrainedTokenizer`"""

        if cfg is None:
            assert token_arr is not None
            assert seq_len_max is not None
        else:
            assert token_arr is None
            assert seq_len_max is None
            # Avoid isinstance() because of circular import
            if type(cfg).__name__ == "ConfigHolder":
                cfg = cfg.dataset_cfg

            seq_len_max = cfg.seq_len_max
            token_arr = cfg.token_arr

        super().__init__(max_len=seq_len_max, **kwargs)
        # We are having to do evil things here
        vocab: dict[str, int] = {token: i for i, token in enumerate(token_arr)}
        vocab[self.unk_token] = len(vocab)
        self.vocab: dict[str, int] = vocab

        self.added_tokens_encoder: dict[str, int] = vocab
        self.added_tokens_decoder: dict[int, str] = {i: token for token, i in vocab.items()}

        self.unique_no_split_tokens = token_arr
        self._create_trie(self.unique_no_split_tokens)

        # IDs specified during construction
        self.bos_token_id: int = self.added_tokens_encoder[self.bos_token]
        self.eos_token_id: int = self.added_tokens_encoder[self.eos_token]
        self.pad_token_id: int = self.added_tokens_encoder[self.pad_token]

    def __call__(self, text, **kwargs) -> BatchEncoding:
        """
        Tokenizer will take a list of strings and encode each
        I.e. a single example should be a continuous string
            "a b c d e f" not ["a", "b", "c", "d", "e", "f"]
        """
        try:
            return super().__call__(text, **kwargs)
        except NotImplementedError as e:
            raise NotImplementedError(
                f"Caught an error during tokenization - probably because you are trying to encode a token not present in the tokenizer's vocabulary"
            )

    def batch_decode(
        self,
        sequences: list[int] | list[list[int]] | ATensor,
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> list[str]:
        if isinstance(sequences, torch.Tensor) and sequences.ndim == 1:
            # Because the slow tokenizer behaves differently to fast ones...
            sequences = sequences.unsqueeze(-1)
        return super().batch_decode(sequences, skip_special_tokens, **kwargs)

    def to_ascii(
        self, sequence: list[int | str] | ATensor, start=None, end=None
    ) -> NDArray:
        # Sequence should be a single maze (not batch)
        if isinstance(sequence, list) and isinstance(sequence[0], str):
            str_sequence = sequence  # already decoded
        else:
            # remove padding
            sequence = torch.tensor(sequence)
            assert sequence.ndim == 1, f"Expected 1D sequence, got {sequence.ndim}D"
            sequence = sequence[sequence != self.pad_token_id]
            str_sequence = self.batch_decode(sequence)

        # Filter out the adjacency list
        str_sequence = str_sequence[
            1 : str_sequence.index(SPECIAL_TOKENS["adjlist_end"])
        ]

        lattice_maze = LatticeMaze.from_tokens(str_sequence)
        return lattice_maze.as_ascii(start=start, end=end)
