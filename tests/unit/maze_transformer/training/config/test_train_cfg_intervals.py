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


def _plus_minus_proportion(
    value: float, proportion: float = 0.1
) -> tuple[float, float]:
    return (
        value * (1 - proportion),
        value * (1 + proportion),
    )


def _in_interval(value: float, interval: tuple[float, float]) -> bool:
    return interval[0] <= value <= interval[1]


def test_get_intervals_with_custom_counts_approx():
    # inputs
    dataset_n_samples: int = 100_000
    batch_size: int = 5
    intervals_count = {
        "print_loss": 1000,
        "checkpoint": 10,
        "eval_fast": 100,
        "eval_slow": 20,
    }
    # expected result
    intervals_expected = {
        "print_loss": _plus_minus_proportion(100),
        "checkpoint": _plus_minus_proportion(10_000),
        "eval_fast": _plus_minus_proportion(1000),
        "eval_slow": _plus_minus_proportion(5000),
    }
    intervals_expected_batched = {
        "print_loss": _plus_minus_proportion(20),
        "checkpoint": _plus_minus_proportion(2_000),
        "eval_fast": _plus_minus_proportion(200),
        "eval_slow": _plus_minus_proportion(1000),
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
        for k, v in calculated_intervals.items():
            assert _in_interval(v, intervals_expected[k])

        calculated_intervals_batched = config.get_intervals(
            dataset_n_samples, mod_batch_size=True, use_defaults_if_missing=use_defaults
        )
        assert isinstance(calculated_intervals_batched, dict)
        for k, v in calculated_intervals_batched.items():
            assert _in_interval(v, intervals_expected_batched[k])


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
        "eval_fast": float("inf"),
        "eval_slow": float("inf"),
    }
    intervals_expected_batched = {
        "print_loss": 10,
        "checkpoint": 4,
        "eval_fast": float("inf"),
        "eval_slow": float("inf"),
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
