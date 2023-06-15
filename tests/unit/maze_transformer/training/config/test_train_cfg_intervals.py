import itertools

import pytest
from torch.optim import RMSprop

from maze_transformer.training.config import _DEFAULT_INTERVAL_COUNTS, TrainConfig


def test_get_intervals_with_default_values():
    n_samples: int = 100
    config = TrainConfig(
        name="test", optimizer=RMSprop, optimizer_kwargs={"lr": 0.001}, batch_size=32
    )
    intervals = config.get_intervals(
        n_samples, use_defaults_if_missing=True, mod_batch_size=False
    )
    assert isinstance(intervals, dict)
    default_counts: dict[str, int] = _DEFAULT_INTERVAL_COUNTS()
    for k, v in intervals.items():
        assert isinstance(k, str)
        assert isinstance(v, int)
        assert v > 0
        assert abs(v - n_samples // default_counts[k]) <= 1


def test_get_intervals_with_custom_intervals():
    # inputs
    batch_size: int = 5
    intervals = {"print_loss": 5, "checkpoint": 20, "eval_fast": 10, "eval_slow": 40}
    # expected result
    intervals_mod_batch_size = {
        "print_loss": 1,
        "checkpoint": 4,
        "eval_fast": 2,
        "eval_slow": 8,
    }

    config = TrainConfig(
        name="test",
        optimizer=RMSprop,
        optimizer_kwargs={"lr": 0.001},
        batch_size=batch_size,
        intervals=intervals,
    )

    for dataset_n_samples, use_defaults in itertools.product(
        [100, None], [True, False]
    ):
        calculated_intervals = config.get_intervals(
            dataset_n_samples,
            mod_batch_size=False,
            use_defaults_if_missing=use_defaults,
        )
        assert isinstance(calculated_intervals, dict)
        assert calculated_intervals == intervals

        calculated_intervals_batched = config.get_intervals(
            dataset_n_samples, mod_batch_size=True, use_defaults_if_missing=use_defaults
        )
        assert isinstance(calculated_intervals_batched, dict)
        assert calculated_intervals_batched == intervals_mod_batch_size


def test_get_intervals_with_custom_counts():
    # inputs
    dataset_n_samples: int = 100
    batch_size: int = 5
    intervals_count = {
        "print_loss": 2,
        "checkpoint": 5,
        "eval_fast": 4,
        "eval_slow": 10,
    }
    # expected result
    intervals_expected = {
        "print_loss": 50,
        "checkpoint": 20,
        "eval_fast": 25,
        "eval_slow": 10,
    }
    intervals_expected_batched = {
        "print_loss": 10,
        "checkpoint": 4,
        "eval_fast": 5,
        "eval_slow": 2,
    }

    config = TrainConfig(
        name="test",
        optimizer=RMSprop,
        optimizer_kwargs={"lr": 0.001},
        batch_size=batch_size,
        intervals_count=intervals_count,
    )

    for use_defaults in [True, False]:
        calculated_intervals = config.get_intervals(
            dataset_n_samples,
            mod_batch_size=False,
            use_defaults_if_missing=use_defaults,
        )
        assert isinstance(calculated_intervals, dict)
        assert calculated_intervals == intervals_expected

        calculated_intervals_batched = config.get_intervals(
            dataset_n_samples, mod_batch_size=True, use_defaults_if_missing=use_defaults
        )
        assert isinstance(calculated_intervals_batched, dict)
        assert calculated_intervals_batched == intervals_expected_batched


def test_get_intervals_raises_with_missing_values():
    config = TrainConfig(
        name="test", optimizer=RMSprop, optimizer_kwargs={"lr": 0.001}, batch_size=32
    )
    with pytest.raises(ValueError):
        config.get_intervals(None, use_defaults_if_missing=False)


def test_get_intervals_raises_with_missing_counts_and_dataset_size():
    intervals_count = {
        "print_loss": 2,
        "checkpoint": 5,
        "eval_fast": 4,
        "eval_slow": 10,
    }
    config = TrainConfig(
        name="test",
        optimizer=RMSprop,
        optimizer_kwargs={"lr": 0.001},
        batch_size=32,
        intervals_count=intervals_count,
    )
    with pytest.raises(ValueError):
        config.get_intervals(None)


def test_get_intervals_with_no_mod_batch_size():
    intervals = {"print_loss": 5, "checkpoint": 20, "eval_fast": 10, "eval_slow": 40}
    config = TrainConfig(
        name="test",
        optimizer=RMSprop,
        optimizer_kwargs={"lr": 0.001},
        batch_size=32,
        intervals=intervals,
    )
    calculated_intervals = config.get_intervals(100, mod_batch_size=False)
    assert calculated_intervals == intervals

def test_get_intervals_disabled_evals():
    # inputs
    dataset_n_samples: int = 100
    batch_size: int = 5
    intervals_count = {
        "print_loss": 2,
        "checkpoint": 5,
        "eval_fast": 0,
        "eval_slow": 0,
    }
    # expected result
    intervals_expected = {
        "print_loss": 50,
        "checkpoint": 20,
        "eval_fast": 101,
        "eval_slow": 101,
    }
    intervals_expected_batched = {
        "print_loss": 10,
        "checkpoint": 4,
        "eval_fast": 21,
        "eval_slow": 21,
    }

    config = TrainConfig(
        name="test",
        optimizer=RMSprop,
        optimizer_kwargs={"lr": 0.001},
        batch_size=batch_size,
        intervals_count=intervals_count,
    )

    for use_defaults in [True, False]:
        calculated_intervals = config.get_intervals(
            dataset_n_samples,
            mod_batch_size=False,
            use_defaults_if_missing=use_defaults,
        )
        assert isinstance(calculated_intervals, dict)
        assert calculated_intervals == intervals_expected

        calculated_intervals_batched = config.get_intervals(
            dataset_n_samples, mod_batch_size=True, use_defaults_if_missing=use_defaults
        )
        assert isinstance(calculated_intervals_batched, dict)
        assert calculated_intervals_batched == intervals_expected_batched