"""
Test that the wrapped tokenizer loaded provides the same outputs
as the original tokenizer (i.e. just using the token map in cfg)

We may want a separate set of tests for different tokenization schemes
"""
import torch
from pytest import mark, param
from transformer_lens import HookedTransformer, HookedTransformerConfig

from maze_transformer.training.config import ConfigHolder
from maze_transformer.training.mazedataset import MazeDatasetConfig
from maze_transformer.training.tokenizer import HuggingMazeTokenizer
from scripts.create_dataset import generate_MazeTokenizer


def test_tokenization_encoding():
    # Check that wrapped tokenizer __call__ returns the same as original tokenizer
    maze = generate_MazeTokenizer(None, 3)

    # Need to generate a config to extract the token map >.<
    # TODO: Part of https://github.com/AISC-understanding-search/maze-transformer/issues/77
    cfg = MazeDatasetConfig(name="testing_maze", grid_n=3, n_mazes=1)
    node_token_map = cfg.node_token_map

    # Adjacency List Tokenization
    maze_str_tokens = maze.as_tokens(node_token_map)

    # Manual Tokenization
    vocab_map = {token: i for i, token in enumerate(cfg.token_arr)}
    maze_tokens = [vocab_map[token] for token in maze_str_tokens]

    # WrappedTokenizer
    # Initialized with a configholder - tokenizer will eventually be a string
    cfg_holder = ConfigHolder(
        train_cfg=None, dataset_cfg=cfg, model_cfg=None, tokenizer=None
    )
    tokenizer = HuggingMazeTokenizer(cfg_holder)

    tokenizer_out = tokenizer(maze_str_tokens)["input_ids"]
    assert torch.all(
        torch.tensor(tokenizer_out).flatten() == torch.tensor(maze_tokens)
    ), "Tokenization mismatch"


def test_to_ascii():
    # Check that the ascii encoding works for multiple different inputs
    maze_str_tokens = """<ADJLIST_START> (1,1) <--> (2,1) ; (2,0) <--> (1,0) ; (0,1) <--> (0,0) ;
    (2,2) <--> (2,1) ; (2,0) <--> (2,1) ; (0,2) <--> (1,2) ; (0,0) <--> (1,0) ; (0,2) <--> (0,1) ;
    <ADJLIST_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (0,0) (1,0) (2,0) (2,1) <PATH_END>""".split()

    target = [
        "#######",
        "#     #",
        "# ### #",
        "# # # #",
        "# # ###",
        "#     #",
        "#######",
    ]

    # Need to generate a config to extract the token map >.<
    cfg = MazeDatasetConfig(name="testing_maze", grid_n=5, n_mazes=1)
    cfg_holder = ConfigHolder(
        train_cfg=None, dataset_cfg=cfg, model_cfg=None, tokenizer=None
    )
    tokenizer = HuggingMazeTokenizer(cfg_holder)

    # Try with string tokens
    assert (
        tokenizer.to_ascii(maze_str_tokens).splitlines() == target
    ), "ASCII encoding from string tokens failed"

    # And with token ids
    token_ids = tokenizer.encode(maze_str_tokens)
    assert (
        tokenizer.to_ascii(token_ids).splitlines() == target
    ), "ASCII encoding from token ids failed"


def test_inside_hooked_transformer():
    # Need to generate a config to extract the token map >.<
    cfg = MazeDatasetConfig(name="testing_maze", grid_n=3, n_mazes=1)

    # Adjacency List Tokenization
    maze_str_tokens = """<ADJLIST_START> (1,1) <--> (2,1) ; (2,0) <--> (1,0) ; (0,1) <--> (0,0) ; 
    (2,2) <--> (2,1) ; (2,0) <--> (2,1) ; (0,2) <--> (1,2) ; (0,0) <--> (1,0) ; (0,2) <--> (0,1) ;
    <ADJLIST_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (0,0) (1,0) (2,0) (2,1) <PATH_END>""".split()

    cfg_holder = ConfigHolder(
        train_cfg=None, dataset_cfg=cfg, model_cfg=None, tokenizer=None
    )
    tokenizer = HuggingMazeTokenizer(cfg_holder)

    hooked_transformer_cfg = HookedTransformerConfig(
        act_fn="relu",
        d_model=5,
        d_head=1,
        n_layers=1,
        n_ctx=100,  # context size
        d_vocab=tokenizer.vocab_size,
    )
    hktransformer = HookedTransformer(cfg=hooked_transformer_cfg, tokenizer=tokenizer)
    token_ids = hktransformer.to_tokens("".join(maze_str_tokens), prepend_bos=False)

    # -- Test Simple Tokenization --
    # Manual Tokenization
    vocab_map = {k: v for v, k in enumerate(cfg.token_arr)}
    maze_tokens = [vocab_map[token] for token in maze_str_tokens]

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
    (0,1) <--> (0,0) ; (0,2) <--> (0,1) ; <ADJLIST_END> <TARGET_START> (1,0) <TARGET_END> 
    <PATH_START> (0,0) (1,0) <PATH_END>""".split()

    batched_tokens = [" ".join(maze_str_tokens), " ".join(maze_str_tokens_2)]

    # Manual Tokenization
    padded_str_2 = ["<PADDING>"] * (
        len(maze_str_tokens) - len(maze_str_tokens_2)
    ) + maze_str_tokens_2
    maze_tokens_2 = [vocab_map[token] for token in padded_str_2]
    batched_tokens_manual = [maze_tokens, maze_tokens_2]

    # WrappedTokenizer use
    token_ids_2 = hktransformer.to_tokens(batched_tokens, prepend_bos=False)

    assert torch.all(
        token_ids_2.cpu() == torch.tensor(batched_tokens_manual)
    ), "Batched tokenization encoding inside HookedTransformer failed"


# Padding Tests
PAD_PLACEHOLDER = -1
test_data = [
    param([1, 2, 3], [PAD_PLACEHOLDER, PAD_PLACEHOLDER, 1, 2, 3], id="short"),
    param([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], id="max_length"),
    param([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6], id="too_long"),
]


@mark.parametrize("inp,expected", test_data)
def test_pad_sequence_param(inp, expected):
    # Initialized with a configholder - tokenizer will eventually be a string
    cfg = MazeDatasetConfig(name="testing_maze", grid_n=3, n_mazes=1)
    cfg_holder = ConfigHolder(
        train_cfg=None, dataset_cfg=cfg, model_cfg=None, tokenizer=None
    )
    tokenizer = HuggingMazeTokenizer(cfg_holder)

    # Pad token id is chosen when the tokenizer is initialized
    expected = [x if x != PAD_PLACEHOLDER else tokenizer.pad_token_id for x in expected]

    # Need to go to string representation to pad
    inp = tokenizer.decode(inp)
    result = tokenizer(
        inp, padding="max_length", truncation=True, max_length=5, return_tensors="pt"
    )["input_ids"][0]

    assert torch.equal(result, torch.tensor(expected))
