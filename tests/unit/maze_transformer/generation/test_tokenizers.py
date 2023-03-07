"""
Test that the wrapped tokenizer loaded provides the same outputs
as the original tokenizer (i.e. just using the token map in cfg)

We may want a separate set of tests for different tokenization schemes
"""
from maze_transformer.training.tokenizer import MazeTokenizer, HuggingMazeTokenizer
from maze_transformer.training.mazedataset import MazeDatasetConfig
from maze_transformer.training.config import ConfigHolder
from scripts.create_dataset import generate_MazeTokenizer

def test_tokenization_encoding():
    # Check that wrapped tokenizer __call__ returns the same as original tokenizer
    maze = generate_MazeTokenizer(None, 3, (0, 0), (2, 1))

    # Need to generate a config to extract the token map >.<
    cfg = MazeDatasetConfig(name='testing_maze', grid_n = 3, n_mazes=1)
    node_token_map = cfg.node_token_map
    
    # Adjacency List Tokenization  
    maze_str_tokens = maze.as_tokens(node_token_map) 

    # Manual Tokenization
    vocab_map = {k: v for v, k in enumerate(cfg.token_arr)}
    maze_tokens = [vocab_map[token] for token in maze_str_tokens]
    
    # WrappedTokenizer
    # Initialized with a configholder - tokenizer will eventually be a string
    cfg_holder = ConfigHolder(train_cfg=None, dataset_cfg=cfg, model_cfg=None, tokenizer=None)
    tokenizer = HuggingMazeTokenizer(cfg_holder)
    print(tokenizer(maze_str_tokens))


def test_ascii_encoding():
    # Check that the ascii encoding works for multiple different inputs
    pass

def test_tokenization_decoding():
    # Check that wrapped tokenizer decode returns the same as iterating through token map
    pass

def test_padding():
    # Check that wrapped tokenizer pad returns the same as original padding function 
    pass

def test_inside_hooked_transformer():
    # Check that the wrapped tokenizer facilitates all HookedTransformer tokenizer-dependent functionality
    maze_id_tokens = tokenizer.to_tokens(maze_str_tokens)
    maze_id_tokens
    pass


test_tokenization_encoding()