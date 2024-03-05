"""
Test that the wrapped tokenizer loaded provides the same outputs
as the original tokenizer (i.e. just using the token map in cfg)

We may want a separate set of tests for different tokenization schemes
"""

from collections import Counter
from itertools import product

import torch
from maze_dataset import MazeDatasetConfig, SolvedMaze
from maze_dataset.generation import get_maze_with_solution
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode
from pytest import mark, param
from transformer_lens import HookedTransformer

from maze_transformer.training.config import BaseGPTConfig, ConfigHolder


@mark.parametrize(
    "tok_mode,grid_size,grid_size_max",
    [
        param(
            tok_mode,
            grid_size,
            grid_size_max,
            id=f"{tok_mode.name.split('_')[-1]},g{grid_size},m{grid_size_max}",
        )
        for tok_mode, grid_size, grid_size_max in product(
            TokenizationMode, [3, 4], [3, 4, 5, 6, 10, 50]
        )
    ],
)
def test_tokenization_encoding(
    tok_mode: TokenizationMode, grid_size: int, grid_size_max: int
):
    # create maze and tokenizer
    solved_maze: SolvedMaze = get_maze_with_solution("gen_dfs", (3, 3))
    tok: MazeTokenizer = MazeTokenizer(
        tokenization_mode=tok_mode, max_grid_size=grid_size
    )

    # convert to strings
    maze_str_tokens: list[str] = solved_maze.as_tokens(tok)

    # cant tokenize if grid size is too big
    if grid_size > grid_size_max:
        tok.encode(maze_str_tokens)
        return

    # check that tokenizer map is as expected
    token_to_index: dict[str, int] = {token: i for i, token in enumerate(tok.token_arr)}
    assert token_to_index == tok.tokenizer_map, "Tokenization mismatch"

    # round trip tokenize
    maze_tokens: list[int] = tok.encode(maze_str_tokens)
    assert maze_str_tokens == tok.decode(maze_tokens), "Tokenization mismatch"

    # create and test actual HuggingMazeTokenizer
    cfg_holder: ConfigHolder = ConfigHolder(
        train_cfg=None,
        model_cfg=None,
        dataset_cfg=MazeDatasetConfig(name="testing_maze", grid_n=3, n_mazes=1),
        maze_tokenizer=tok,
    )

    tokenizer_out = cfg_holder.tokenizer(maze_str_tokens)["input_ids"]
    assert torch.all(
        torch.tensor(tokenizer_out).flatten() == torch.tensor(maze_tokens)
    ), "Tokenization mismatch"


@mark.parametrize(
    "tok_mode",
    [
        param(tok_mode, id=tok_mode.name)
        for tok_mode in [
            TokenizationMode.AOTP_UT_uniform,
            TokenizationMode.AOTP_UT_rasterized,
        ]
    ],
)
def test_to_ascii(tok_mode):
    # Check that the ascii encoding works for multiple different inputs
    maze_str_tokens: list[str] = (
        """<ADJLIST_START> (1,1) <--> (2,1) ; (2,0) <--> (1,0) ; (0,1) <--> (0,0) ;
    (2,2) <--> (2,1) ; (2,0) <--> (2,1) ; (0,2) <--> (1,2) ; (0,0) <--> (1,0) ; (0,2) <--> (0,1) ;
    <ADJLIST_END> <ORIGIN_START> (0,0) <ORIGIN_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (0,0) (1,0) (2,0) (2,1) <PATH_END>""".split()
    )

    target: list[str] = [
        "#######",
        "#S    #",
        "#X### #",
        "#X# # #",
        "#X# ###",
        "#XXE  #",
        "#######",
    ]

    # Need to generate a config to extract the token map >.<
    maze_tok_cfg = MazeTokenizer(tokenization_mode=tok_mode)
    cfg_holder = ConfigHolder(
        train_cfg=None,
        dataset_cfg=MazeDatasetConfig(name="testing_maze", grid_n=5, n_mazes=1),
        model_cfg=None,
        maze_tokenizer=maze_tok_cfg,
    )

    # Try with string tokens
    assert (
        cfg_holder.tokenizer.to_ascii(maze_str_tokens).splitlines() == target
    ), "ASCII encoding from string tokens failed"
    # And with token ids
    token_ids = cfg_holder.tokenizer.encode(maze_str_tokens)
    assert (
        cfg_holder.tokenizer.to_ascii(token_ids).splitlines() == target
    ), "ASCII encoding from token ids failed"


_ASCII_MAZES: dict[str, tuple[str, list[str]]] = dict(
    small_3x3 = (
        "<ADJLIST_START> (2,0) <--> (2,1) ; (0,0) <--> (0,1) ; (0,0) <--> (1,0) ; (0,2) <--> (1,2) ; (1,0) <--> (2,0) ; (0,2) <--> (0,1) ; (2,2) <--> (2,1) ; (1,1) <--> (2,1) ; <ADJLIST_END> <ORIGIN_START> (0,0) <ORIGIN_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (0,0) (1,0) (2,0) (2,1) <PATH_END>",
        [
            "#######",
            "#S    #",
            "#X### #",
            "#X# # #",
            "#X# ###",
            "#XXE  #",
            "#######",
        ],
    ),
    big_10x10 = (
        "<ADJLIST_START> (8,2) <--> (8,3) ; (3,7) <--> (3,6) ; (6,7) <--> (6,8) ; (4,6) <--> (5,6) ; (9,5) <--> (9,4) ; (3,3) <--> (3,4) ; (5,1) <--> (4,1) ; (2,6) <--> (2,7) ; (8,5) <--> (8,4) ; (1,9) <--> (2,9) ; (4,1) <--> (4,2) ; (0,8) <--> (0,7) ; (5,4) <--> (5,3) ; (6,3) <--> (6,4) ; (5,0) <--> (4,0) ; (5,3) <--> (5,2) ; (3,1) <--> (2,1) ; (9,1) <--> (9,0) ; (3,5) <--> (3,6) ; (5,5) <--> (6,5) ; (7,1) <--> (7,2) ; (0,1) <--> (1,1) ; (7,8) <--> (8,8) ; (3,9) <--> (4,9) ; (4,6) <--> (4,7) ; (0,6) <--> (0,7) ; (3,4) <--> (3,5) ; (6,0) <--> (5,0) ; (7,7) <--> (7,6) ; (1,6) <--> (0,6) ; (6,1) <--> (6,0) ; (8,6) <--> (8,7) ; (9,9) <--> (9,8) ; (1,8) <--> (1,9) ; (2,1) <--> (2,2) ; (9,2) <--> (9,3) ; (5,9) <--> (6,9) ; (3,2) <--> (2,2) ; (0,8) <--> (0,9) ; (5,6) <--> (5,7) ; (2,3) <--> (2,4) ; (4,5) <--> (4,4) ; (8,9) <--> (8,8) ; (9,6) <--> (8,6) ; (3,7) <--> (3,8) ; (8,0) <--> (7,0) ; (6,1) <--> (6,2) ; (0,1) <--> (0,0) ; (7,3) <--> (7,4) ; (9,4) <--> (9,3) ; (9,6) <--> (9,5) ; (8,7) <--> (7,7) ; (5,2) <--> (5,1) ; (0,0) <--> (1,0) ; (7,2) <--> (7,3) ; (2,5) <--> (2,6) ; (4,9) <--> (5,9) ; (5,5) <--> (5,4) ; (5,6) <--> (6,6) ; (7,8) <--> (7,9) ; (1,7) <--> (2,7) ; (4,6) <--> (4,5) ; (1,1) <--> (1,2) ; (3,1) <--> (3,0) ; (1,5) <--> (1,6) ; (8,3) <--> (8,4) ; (9,9) <--> (8,9) ; (8,5) <--> (7,5) ; (1,4) <--> (2,4) ; (3,0) <--> (4,0) ; (3,3) <--> (4,3) ; (6,9) <--> (6,8) ; (1,0) <--> (2,0) ; (6,0) <--> (7,0) ; (8,0) <--> (9,0) ; (2,3) <--> (2,2) ; (2,8) <--> (3,8) ; (5,7) <--> (6,7) ; (1,3) <--> (0,3) ; (9,7) <--> (9,8) ; (7,5) <--> (7,4) ; (1,8) <--> (2,8) ; (6,5) <--> (6,4) ; (0,2) <--> (1,2) ; (0,7) <--> (1,7) ; (0,3) <--> (0,2) ; (4,3) <--> (4,2) ; (5,8) <--> (4,8) ; (9,1) <--> (8,1) ; (9,2) <--> (8,2) ; (1,3) <--> (1,4) ; (2,9) <--> (3,9) ; (4,8) <--> (4,7) ; (0,5) <--> (0,4) ; (8,1) <--> (7,1) ; (0,3) <--> (0,4) ; (9,7) <--> (9,6) ; (7,6) <--> (6,6) ; (1,5) <--> (0,5) ; <ADJLIST_END> <ORIGIN_START> (6,2) <ORIGIN_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (6,2) (6,1) (6,0) (5,0) (4,0) (3,0) (3,1) (2,1) <PATH_END>",
        [
            "#####################",
            "#   #       #       #",
            "# # # # ### # # #####",
            "# #   #   #   # #   #",
            "# ####### ##### # # #",
            "# #E      #     # # #",
            "###X# ########### # #",
            "#XXX# #           # #",
            "#X##### ########### #",
            "#X#     #         # #",
            "#X# ######### ### # #",
            "#X#         #   # # #",
            "#X######### # # ### #",
            "#XXXXS#     # #     #",
            "# ########### #######",
            "# #         #   #   #",
            "# # ####### ### # ###",
            "# # #       #   #   #",
            "# # # ####### ##### #",
            "#   #               #",
            "#####################",
        ],
    ),
)


@mark.parametrize(
    "maze_ascii, tok_mode, tokens",
    [
        param(
            _ASCII_MAZES[maze_ascii_key][1], # maze_ascii
            tok_mode,                        # tok_mode
            _ASCII_MAZES[maze_ascii_key][0], # tokens
            id=f"{tok_mode.name}_{maze_ascii_key}",
        )
        for maze_ascii_key, tok_mode in product(
            ["small_3x3", "big_10x10"], 
            [TokenizationMode.AOTP_UT_uniform, TokenizationMode.AOTP_UT_rasterized],
        )
    ]
)
def test_maze_to_tokens_roundtrip(
        maze_ascii: list[str], 
        tok_mode: TokenizationMode,
        tokens: str,
    ):
    tokens_original_split: list[str] = tokens.split()
    def get_token_regions(toks: list[str]) -> tuple[list[str], list[str]]:
        adj_list_start, adj_list_end = toks.index("<ADJLIST_START>") + 1, tokens.index(
            "<ADJLIST_END>"
        )
        adj_list = toks[adj_list_start:adj_list_end]
        non_adj_list = toks[:adj_list_start] + toks[adj_list_end:]
        return adj_list, non_adj_list

    # join into a single string, and get a maze out
    ascii_str: str = "\n".join(maze_ascii)
    maze: SolvedMaze = SolvedMaze.from_ascii(ascii_str)
    # init tokenizer
    tokenizer: MazeTokenizer = MazeTokenizer(tokenization_mode=tok_mode)

    # maze as tokens
    tokens_from_maze: list[str] = maze.as_tokens(tokenizer)
    adj_list, non_adj_list = get_token_regions(tokens_from_maze)

    # maze round trip
    maze_roundtrip: SolvedMaze = SolvedMaze.from_tokens(tokens_from_maze, tokenizer)
    tokens_roundtrip: list[str] = maze_roundtrip.as_tokens(tokenizer)
    adj_list_rt, non_adj_list_rt = get_token_regions(tokens_roundtrip)

    # regions from original tokens
    adj_list_orig, non_adj_list_orig = get_token_regions(tokens_original_split)


    # check that the maze works
    assert maze == maze_roundtrip

    # check that the counters match
    counter_original: Counter = Counter(tokens_original_split)
    counter_from_maze: Counter = Counter(tokens_from_maze)
    counter_roundtrip: Counter = Counter(tokens_roundtrip)

    assert counter_original == counter_from_maze
    assert counter_original == counter_roundtrip

    # check that the token regions match
    assert non_adj_list_orig == non_adj_list
    assert non_adj_list_rt == non_adj_list
 

@mark.parametrize(
    "tok_mode",
    [
        param(TokenizationMode.AOTP_UT_uniform, id="AOTP_UT_uniform"),
        param(TokenizationMode.AOTP_UT_rasterized, id="AOTP_UT_rasterized"),
    ],
)
def test_tokenizer_inside_hooked_transformer(tok_mode):
    maze_tok_cfg = MazeTokenizer(tokenization_mode=tok_mode)
    cfg_holder = ConfigHolder(
        train_cfg=None,
        dataset_cfg=MazeDatasetConfig(name="testing_maze", grid_n=3, n_mazes=1),
        model_cfg=BaseGPTConfig(
            name="for test_tokenizer_inside_hooked_transformer",
            act_fn="relu",
            d_model=5,
            d_head=1,
            n_layers=1,
        ),
        maze_tokenizer=maze_tok_cfg,
    )

    # Adjacency List Tokenization
    maze_str_tokens = """<ADJLIST_START> (1,1) <--> (2,1) ; (2,0) <--> (1,0) ; (0,1) <--> (0,0) ;
    (2,2) <--> (2,1) ; (2,0) <--> (2,1) ; (0,2) <--> (1,2) ; (0,0) <--> (1,0) ; (0,2) <--> (0,1) ;
    <ADJLIST_END> <ORIGIN_START> (0,0) <ORIGIN_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (0,0) (1,0) (2,0) (2,1) <PATH_END>""".split()

    hktransformer: HookedTransformer = cfg_holder.create_model()

    token_ids = hktransformer.to_tokens(" ".join(maze_str_tokens), prepend_bos=False)
    token_ids_sep = hktransformer.to_tokens(
        maze_str_tokens, prepend_bos=False
    ).flatten()
    assert torch.allclose(token_ids, token_ids_sep), "Tokenization mismatch"

    # -- Test Simple Tokenization --
    # Manual Tokenization
    vocab_map = {k: v for v, k in enumerate(maze_tok_cfg.token_arr)}
    maze_tokens_manual = [vocab_map[token] for token in maze_str_tokens]
    maze_tokens = maze_tok_cfg.encode(maze_str_tokens)
    assert maze_tokens == maze_tokens_manual, "Manual tokenization failed"

    assert torch.all(
        token_ids.flatten().cpu() == torch.tensor(maze_tokens)
    ), "Simple tokenization encoding inside HookedTransformer failed"

    # Test casting back to string tokens
    str_tokens = hktransformer.to_str_tokens(token_ids)
    assert (
        str_tokens == maze_str_tokens
    ), "Simple tokenization decoding inside HookedTransformer failed"

    # -- Test Batched Tokenization --
    maze_str_tokens_2 = """<ADJLIST_START> (1,1) <--> (2,1) ; (2,0) <--> (1,0) ;
    (0,1) <--> (0,0) ; (0,2) <--> (0,1) ; <ADJLIST_END> <ORIGIN_START> (0,0) <ORIGIN_END> <TARGET_START> (1,0) <TARGET_END>
    <PATH_START> (0,0) (1,0) <PATH_END>""".split()

    total_len: int = max(len(maze_str_tokens), len(maze_str_tokens_2))

    # Manual Tokenization
    padded_str_2 = ["<PADDING>"] * (
        total_len - len(maze_str_tokens_2)
    ) + maze_str_tokens_2
    print(f"{len(padded_str_2) = }")
    print(f"{len(maze_tokens) = }")
    batched_tokens = [" ".join(maze_str_tokens), " ".join(padded_str_2)]
    maze_tokens_2 = [vocab_map[token] for token in padded_str_2]
    print(f"{len(maze_tokens_2) = }")
    batched_tokens_manual = [maze_tokens, maze_tokens_2]

    # WrappedTokenizer use
    token_ids_2 = hktransformer.to_tokens(batched_tokens, prepend_bos=False).cpu()
    batched_tokens_manual_tensor = torch.tensor(batched_tokens_manual)

    manual_hk_match = token_ids_2 == batched_tokens_manual_tensor
    if not manual_hk_match.all():
        raise AssertionError(
            f"Batched tokenization encoding inside HookedTransformer failed",
            f"{manual_hk_match.shape = }, {token_ids_2.shape = }, {batched_tokens_manual_tensor.shape = }",
            f"{[len(x.split(' ')) for x in batched_tokens] = }",
            f"{batched_tokens = }" f"{manual_hk_match = }",
            f"{token_ids_2 = }",
            f"{batched_tokens_manual_tensor = }",
        )


# Padding Tests
PAD_PLACEHOLDER = -1


@mark.parametrize(
    "inp,expected,tok_mode",
    [
        param(
            [1, 2, 3],
            [PAD_PLACEHOLDER, PAD_PLACEHOLDER, 1, 2, 3],
            TokenizationMode.AOTP_UT_uniform,
            id="short+uniform",
        ),
        param(
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            TokenizationMode.AOTP_UT_uniform,
            id="max_length+uniform",
        ),
        param(
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6],
            TokenizationMode.AOTP_UT_uniform,
            id="too_long+uniform",
        ),
        param(
            [1, 2, 3],
            [PAD_PLACEHOLDER, PAD_PLACEHOLDER, 1, 2, 3],
            TokenizationMode.AOTP_UT_rasterized,
            id="short+rasterized",
        ),
        param(
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            TokenizationMode.AOTP_UT_rasterized,
            id="max_length+rasterized",
        ),
        param(
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6],
            TokenizationMode.AOTP_UT_rasterized,
            id="too_long+rasterized",
        ),
    ],
)
def test_pad_sequence_param(inp, expected, tok_mode):
    maze_tok_cfg: MazeTokenizer = MazeTokenizer(tokenization_mode=tok_mode)
    cfg_holder: ConfigHolder = ConfigHolder(
        train_cfg=None,
        dataset_cfg=MazeDatasetConfig(name="testing_maze", grid_n=3, n_mazes=1),
        model_cfg=None,
        maze_tokenizer=maze_tok_cfg,
    )

    # Pad token id is chosen when the tokenizer is initialized
    expected = [
        x if x != PAD_PLACEHOLDER else cfg_holder.tokenizer.pad_token_id
        for x in expected
    ]

    # Need to go to string representation to pad
    inp = cfg_holder.tokenizer.decode(inp)
    result = cfg_holder.tokenizer(
        inp, padding="max_length", truncation=True, max_length=5, return_tensors="pt"
    )["input_ids"][0]

    assert torch.equal(result, torch.tensor(expected))


# def test_manual_tokenizer():
#     """tests setting the kwargs for a pretrained tokenizer, instead of getting HuggingMazeTokenizer

#     this is mostly just testing to make sure it doesnt crash lol
#     """

#     cfg: ConfigHolder = ConfigHolder(
#         train_cfg=None,
#         dataset_cfg=MazeDatasetConfig(name="testing_maze", grid_n=3, n_mazes=1),
#         model_cfg=None,
#         pretrainedtokenizer_kwargs=dict(
#             bos_token="<bos>",
#             eos_token="<eos>",
#             pad_token="<pad>",
#         ),
#     )

#     tok = cfg.tokenizer

#     assert tok.bos_token == "<bos>"
#     assert tok.eos_token == "<eos>"
#     assert tok.pad_token == "<pad>"
