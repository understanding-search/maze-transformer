from itertools import chain

# Avoid circular import from training/config.py
from typing import TYPE_CHECKING, Union  # need Union as "a" | "b" doesn't work

import torch
from muutils.tensor_utils import ATensor, NDArray
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import BatchEncoding

from maze_transformer.generation.constants import SPECIAL_TOKENS, Coord, CoordTup
from maze_transformer.generation.latticemaze import LatticeMaze

if TYPE_CHECKING:
    from maze_transformer.training.config import ConfigHolder, MazeDatasetConfig
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
                for c_s, c_e in maze.as_adjlist()
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
        self, cfg: Union["ConfigHolder", "MazeDatasetConfig"], **kwargs
    ) -> None:
        # Avoid isinstance() because of circular import
        if type(cfg).__name__ == "ConfigHolder":
            cfg = cfg.dataset_cfg

        super().__init__(max_len=cfg.seq_len_max, **kwargs)
        # We are having to do evil things here
        vocab = {token: i for i, token in enumerate(cfg.token_arr)}
        vocab[self.unk_token] = len(vocab)
        self.vocab = vocab

        self.added_tokens_encoder = vocab
        self.added_tokens_decoder = {i: token for token, i in vocab.items()}

        self.unique_no_split_tokens = cfg.token_arr
        self._create_trie(self.unique_no_split_tokens)

        # IDs specified during construction
        self.bos_token_id = self.added_tokens_encoder[self.bos_token]
        self.eos_token_id = self.added_tokens_encoder[self.eos_token]
        self.pad_token_id = self.added_tokens_encoder[self.pad_token]

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
