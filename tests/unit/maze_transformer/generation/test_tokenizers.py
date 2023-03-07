"""
Test that the wrapped tokenizer loaded provides the same outputs
as the original tokenizer (i.e. just using the token map in cfg)

We may want a separate set of tests for different tokenization schemes
"""
from maze_transformer.training.tokenizer import MazeTokenizer, HuggingMazeTokenizer
from maze_transformer.training.mazedataset import MazeDatasetConfig
from maze_transformer.training.config import ConfigHolder
from scripts.create_dataset import generate_MazeTokenizer

import torch 
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
    
    # tokenizer_out = tokenizer(maze_str_tokens, padding='')['input_ids']
    print(maze_str_tokens)
    tokenizer_out = tokenizer([maze_str_tokens, [0]])['input_ids']
    print(tokenizer_out) 
    # torch.all(tokenizer_out.flatten() == torch.tensor(maze_tokens))


def test_ascii_encoding():
    # Check that the ascii encoding works for multiple different inputs
    pass

def test_tokenization_decoding():
    # Check that wrapped tokenizer decode returns the same as iterating through token map
    pass

def test_padding():
    # Check that wrapped tokenizer pad returns the same as original padding function 
    pass


from transformer_lens import HookedTransformer, HookedTransformerConfig
def test_inside_hooked_transformer():
    # Check that the wrapped tokenizer facilitates all HookedTransformer tokenizer-dependent functionality
    maze = generate_MazeTokenizer(None, 3, (0, 0), (2, 1))

    # Need to generate a config to extract the token map >.<
    cfg = MazeDatasetConfig(name='testing_maze', grid_n = 3, n_mazes=1)
    node_token_map = cfg.node_token_map
    
    # Adjacency List Tokenization  
    maze_str_tokens = maze.as_tokens(node_token_map) 

    cfg_holder = ConfigHolder(train_cfg=None, dataset_cfg=cfg, model_cfg=None, tokenizer=None)
    tokenizer = HuggingMazeTokenizer(cfg_holder)
    
    hooked_transformer_cfg = HookedTransformerConfig(
            act_fn='relu',
            d_model=5,
            d_head=1,
            n_layers=1,
            n_ctx=100, # context size
            d_vocab=tokenizer.vocab_size
    )
    hktransformer = HookedTransformer(cfg=hooked_transformer_cfg, tokenizer=tokenizer)

    print(maze_str_tokens)
    token_ids = hktransformer.to_tokens(''.join(maze_str_tokens))
    print(token_ids)
    str_tokens = hktransformer.to_str_tokens(token_ids)
    print(str_tokens)
    

# test_tokenization_encoding()
test_inside_hooked_transformer()