from dataclasses import dataclass, field
from itertools import chain

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

from transformers import PreTrainedTokenizer
from muutils.tensor_utils import ATensor, NDArray
from maze_transformer.training.config import ConfigHolder
from maze_transformer.generation.latticemaze import SPECIAL_TOKENS
# from maze_transformer.training.mazedataset import MazeDatasetConfig

class HuggingMazeTokenizer(PreTrainedTokenizer):
    vocab: dict[str, int]# map of token_ids to strings

    bos_token: str = SPECIAL_TOKENS["padding"]
    eos_token: str = SPECIAL_TOKENS["padding"] 
    unk_token: str = '<WOOPS>'
    #! pad_token is set to tokenizer.eos_token by TransformerLens
    # TODO check if this is a problem for us
    pad_token: str = eos_token
    additional_special_tokens: list[str] = [x for x in SPECIAL_TOKENS.values() if x not in [SPECIAL_TOKENS['padding']]]
    
    # IDs specified during construction
    # bos_token_id: int
    # eos_token_id: int
    # pad_token_id: int

    # Overwrite class attributes
    padding_side = "left"
    truncation_side = "left" #! strange choice, but it's what we did in pad_sequence
    
    name_or_path = "maze_tokenizer"
    
    def __init__(self, cfg: ConfigHolder, **kwargs):
        super().__init__(max_len=cfg.dataset_cfg.seq_len_max, **kwargs)
        self._add_tokens()
        vocab = {k: v for v, k in enumerate(cfg.dataset_cfg.token_arr)}
        self.added_tokens_encoder = vocab
        self.added_tokens_decoder = {v: k for k, v in vocab.items()}
        self.vocab = vocab #! Feels like this should be automatic through the super().__init__ call
        
        self.bos_token_id = self.vocab[self.bos_token]
        self.eos_token_id = self.vocab[self.eos_token]
        self.pad_token_id = self.vocab[self.pad_token]

    # def __call__(self, str_tokens: list[str], **kwargs) -> ATensor:
    #     tokens = [self.vocab[token] for token in maze.as_tokens(cfg.node_token_map)]
    #     maze_tokens = self.encode(tokens, **kwargs)
    #     return maze_tokens

    # def batch_decode(self, sequences: list[int] | list[list[int]] | ATensor, skip_special_tokens: bool = False, **kwargs) -> list[str]:
    #     return super().batch_decode(sequences, skip_special_tokens, clean_up_tokenization_spaces=False, **kwargs)

    # def decode(self, sequence: list[int] | ATensor, skip_special_tokens: bool = False, **kwargs) -> str:
    #     return super().decode(sequence, skip_special_tokens, clean_up_tokenization_spaces=False, **kwargs)
    
    def to_image(self, sequence: list[int] | ATensor | list[str], **kwargs) -> NDArray:

        if isinstance(sequence, list) and isinstance(sequence[0], str):
            str_sequence = sequence # already decoded
        else:
            str_sequence = self.decode(sequence)

        lattice_maze = LatticeMaze.from_tokens(str_sequence) 
        return lattice_maze.as_img(**kwargs)
   