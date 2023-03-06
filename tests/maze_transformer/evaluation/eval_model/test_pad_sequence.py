import torch
from pytest import mark, param
from transformers import OpenAIGPTConfig

from maze_transformer.evaluation.eval_model import pad_sequence

test_data = [
    param([1, 2, 3], [0, 0, 1, 2, 3], id="short"),
    param([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], id="max_length"),
    param([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6], id="too_long"),
    param([], [0, 0, 0, 0, 0], id="empty"),
]


@mark.parametrize("input,expected", test_data)
def test_pad_sequence_param(input, expected):
    model_cfg = OpenAIGPTConfig(n_positions=5)
    result = pad_sequence(torch.tensor(input), model_cfg)
    assert torch.equal(result, torch.tensor(expected))


def test_pad_sequence_specify_token():
    seq = torch.tensor([1, 2, 3])
    model_cfg = OpenAIGPTConfig(n_positions=5, pad_token_id=-1)
    result = pad_sequence(seq, model_cfg)
    assert torch.equal(result, torch.tensor([-1, -1, 1, 2, 3]))
