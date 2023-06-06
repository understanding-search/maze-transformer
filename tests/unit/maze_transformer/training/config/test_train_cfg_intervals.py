import pytest
import torch
from torch.optim import RMSprop
from maze_transformer.training.config import TrainConfig, _DEFAULT_INTERVAL_COUNTS

def test_get_intervals_with_default_values():
    n_samples: int = 100
    config = TrainConfig(name="test", optimizer=RMSprop, optimizer_kwargs={"lr": 0.001}, batch_size=32)
    intervals = config.get_intervals(n_samples, use_defaults_if_missing=True, mod_batch_size=False)
    assert isinstance(intervals, dict)
    for k,v in intervals.items():
        assert isinstance(k, str)
        assert isinstance(v, int)
        assert v > 0
        assert abs(v - n_samples // _DEFAULT_INTERVAL_COUNTS[k]) <= 1

def test_get_intervals_with_custom_intervals():
    intervals = {"print_loss": 5, "checkpoint": 20, "eval_fast": 10, "eval_slow": 40}
    config = TrainConfig(name="test", optimizer=RMSprop, optimizer_kwargs={"lr": 0.001}, batch_size=32, intervals=intervals)
    assert config.get_intervals(100, mod_batch_size=False) == intervals

def test_get_intervals_with_custom_counts():
    intervals_count = {"print_loss": 2, "checkpoint": 5, "eval_fast": 4, "eval_slow": 10}
    config = TrainConfig(name="test", optimizer=RMSprop, optimizer_kwargs={"lr": 0.001}, batch_size=32, intervals_count=intervals_count)
    calculated_intervals = config.get_intervals(100)
    assert isinstance(calculated_intervals, dict)
    assert calculated_intervals == {"print_loss": 50, "checkpoint": 20, "eval_fast": 25, "eval_slow": 10}

def test_get_intervals_raises_with_missing_values():
    config = TrainConfig(name="test", optimizer=RMSprop, optimizer_kwargs={"lr": 0.001}, batch_size=32)
    with pytest.raises(ValueError):
        config.get_intervals(None, use_defaults_if_missing=False)

def test_get_intervals_raises_with_missing_counts_and_dataset_size():
    intervals_count = {"print_loss": 2, "checkpoint": 5, "eval_fast": 4, "eval_slow": 10}
    config = TrainConfig(name="test", optimizer=RMSprop, optimizer_kwargs={"lr": 0.001}, batch_size=32, intervals_count=intervals_count)
    with pytest.raises(ValueError):
        config.get_intervals(None)

def test_get_intervals_with_no_mod_batch_size():
    intervals = {"print_loss": 5, "checkpoint": 20, "eval_fast": 10, "eval_slow": 40}
    config = TrainConfig(name="test", optimizer=RMSprop, optimizer_kwargs={"lr": 0.001}, batch_size=32, intervals=intervals)
    calculated_intervals = config.get_intervals(100, mod_batch_size=False)
    assert calculated_intervals == intervals
