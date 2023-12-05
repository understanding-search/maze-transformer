import pytest

from maze_transformer.utils.padding import pad_and_batch_tensors


@pytest.mark.parametrize(
    "batch_size,padding_idx,padding_dir,expected_shapes",
    [
        (2, 0, "right", [(2, 3), (2, 4)]),
        (2, -1, "left", [(2, 3), (2, 4)]),
        (1, 0, "right", [(1, 2), (1, 3), (1, 1), (1, 4)]),
        (3, 0, "left", [(3, 3), (1, 4)]),
        (4, 0, "left", [(4, 4)]),
    ],
)
def test_pad_and_batch_tensors(batch_size, padding_idx, padding_dir, expected_shapes):
    contexts_tokens = [[1, 2], [1, 2, 3], [1], [1, 2, 3, 4]]
    padded_batches = pad_and_batch_tensors(
        contexts_tokens, batch_size, padding_idx, padding_dir
    )

    # Check if the number of batches is correct
    assert len(padded_batches) == (len(contexts_tokens) + batch_size - 1) // batch_size

    # Check if each batch has the correct shape
    assert [b.shape == ex_shape for b, ex_shape in zip(padded_batches, expected_shapes)]

    # Check if padding is correctly applied
    for idx_b, batch in enumerate(padded_batches):
        for idx_s, seq in enumerate(batch):
            context: list[int] = contexts_tokens[idx_b * batch_size + idx_s]
            if padding_dir == "right":
                assert seq[: len(context)].tolist() == context
                assert (seq[len(context) :] == padding_idx).all()
            else:
                assert seq[-len(context) :].tolist() == context
                assert (seq[: -len(context)] == padding_idx).all()
