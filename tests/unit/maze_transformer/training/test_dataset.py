import torch

from maze_transformer.training.dataset import IndexedArray


def test_indexed_array_get_len():
    indexed_array = IndexedArray.from_sequences(
        [
            torch.tensor(
                [1, 2, 3, 4, 5],
                dtype=torch.int16,
            )
            for _ in range(10)
        ]
    )

    assert indexed_array.get_len(0) == 5
    assert indexed_array.get_len(9) == 5


def test_indexed_array_get_all_lengths():
    indexed_array = IndexedArray.from_sequences(
        [
            torch.tensor(
                [1, 2, 3, 4, 5],
                dtype=torch.int16,
            )
            for _ in range(10)
        ]
    )

    all_lengths = indexed_array.get_all_lengths()

    assert all_lengths[0] == 5
    assert all_lengths[9] == 5
