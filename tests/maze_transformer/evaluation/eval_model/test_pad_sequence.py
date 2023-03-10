import torch
from pytest import mark, param
from transformers import OpenAIGPTConfig

from maze_transformer.evaluation.eval_model import pad_sequence
from maze_transformer.training.tokenizer import MazeTokenizer, HuggingMazeTokenizer
from maze_transformer.training.mazedataset import MazeDatasetConfig
from maze_transformer.training.config import ConfigHolder

PADDING_ID = 8
test_data = [
    param([1, 2, 3], [PADDING_ID, PADDING_ID, 1, 2, 3], id="short"),
    param([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], id="max_length"),
    param([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6], id="too_long"),
]


@mark.parametrize("inp,expected", test_data)
def test_pad_sequence_param(inp, expected):
    # Initialized with a configholder - tokenizer will eventually be a string
    cfg = MazeDatasetConfig(name='testing_maze', grid_n = 3, n_mazes=1)
    cfg_holder = ConfigHolder(train_cfg=None, dataset_cfg=cfg, model_cfg=None, tokenizer=None)
    tokenizer = HuggingMazeTokenizer(cfg_holder)

    # Need to go to string representation to pad 
    inp = tokenizer.decode(inp)
    result = tokenizer(inp, padding="max_length", truncation=True, max_length=5, return_tensors='pt')['input_ids'][0]
    
    assert torch.equal(result, torch.tensor(expected))