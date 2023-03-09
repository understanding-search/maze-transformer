from dataclasses import dataclass, field
from itertools import chain

from muutils.tensor_utils import ATensor, NDArray
# Avoid circular import from training/config.py
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from maze_transformer.training.config import ConfigHolder

from transformers import PreTrainedTokenizer
import torch

from maze_transformer.generation.latticemaze import (
    SPECIAL_TOKENS,
    CoordArray,
    CoordTup,
    LatticeMaze,
)

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

class HuggingMazeTokenizer(PreTrainedTokenizer):
    vocab: dict[str, int]# map of token_ids to strings

    bos_token: str = SPECIAL_TOKENS["padding"]
    eos_token: str = SPECIAL_TOKENS["padding"] 
    unk_token: str = '<WHOOPS>'
    #! pad_token is set to tokenizer.eos_token by TransformerLens
    # TODO check if this is a problem for us
    pad_token: str = eos_token
    vocab_size: int = 0   
    additional_special_tokens: list[str] = [x for x in SPECIAL_TOKENS.values() if x not in [SPECIAL_TOKENS['padding']]]
    
    # Overwrite class attributes
    padding_side = "left"
    truncation_side = "left" #! strange choice, but it's what we did in pad_sequence
    
    name_or_path = "maze_tokenizer"
    
    def __init__(self, cfg: "ConfigHolder", **kwargs):
        super().__init__(max_len=cfg.dataset_cfg.seq_len_max, **kwargs)
        # We are having to do evil things here
        vocab = {k: v for v, k in enumerate(cfg.dataset_cfg.token_arr)}
        vocab[self.unk_token] = len(vocab)

        self.added_tokens_encoder = vocab 
        self.added_tokens_decoder = {v: k for k, v in vocab.items()}

        self.unique_no_split_tokens = cfg.dataset_cfg.token_arr
        self._create_trie(self.unique_no_split_tokens)
        
        # IDs specified during construction
        self.bos_token_id = self.added_tokens_encoder[self.bos_token]
        self.eos_token_id = self.added_tokens_encoder[self.eos_token] # Because the slow tokenizer behaves differently to fast ones...
        self.pad_token_id = self.added_tokens_encoder[self.pad_token]

    def batch_decode(self, sequences: list[int] | list[list[int]] | ATensor, skip_special_tokens: bool = False, **kwargs) -> list[str]:
        if isinstance(sequences, torch.Tensor) and sequences.ndim == 1:
            sequences = sequences.unsqueeze(-1) # Because the slow tokenizer behaves differently to fast ones...
        return super().batch_decode(sequences, skip_special_tokens, **kwargs)
    
    def to_ascii(self, sequence: list[int | str] | ATensor, start=None, end=None) -> NDArray:
        # Sequence should be a single maze (not batch)
        if isinstance(sequence, list) and isinstance(sequence[0], str):
            str_sequence = sequence # already decoded
        else:
            str_sequence = self.batch_decode(torch.tensor(sequence).unsqueeze(-1))
        
        # Filter out the adjacency list
        str_sequence = str_sequence[1:str_sequence.index(SPECIAL_TOKENS['adjlist_end'])]
        
        lattice_maze = LatticeMaze.from_tokens(str_sequence) 
        return lattice_maze.as_ascii(start=start, end=end)