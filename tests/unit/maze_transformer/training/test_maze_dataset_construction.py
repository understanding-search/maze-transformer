import pytest
from maze_dataset import SPECIAL_TOKENS, MazeDataset, MazeDatasetConfig
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode, MazeTokenizerModular


@pytest.mark.parametrize(
    "tokenizer",
    [
        pytest.param(TokenizationMode.AOTP_UT_rasterized.to_legacy_tokenizer(), id="rasterized"),
        pytest.param(TokenizationMode.AOTP_UT_uniform.to_legacy_tokenizer(), id="uniform"),
        pytest.param(MazeTokenizerModular(), id="MazeTokenizerModular"),
    ],
)
def test_dataset_construction(tokenizer: MazeTokenizer | MazeTokenizerModular):
    config: MazeDatasetConfig = MazeDatasetConfig(
        name="test",
        grid_n=2,
        n_mazes=3,
    )
    dataset: MazeDataset = MazeDataset.from_config(cfg=config)

    # check the tokenization
    test_tokenizations: list[list[str]] = dataset.as_tokens(tokenizer)

    # the adj_list always gets shuffled, so easier to check the paths
    # this will be much simpler once token utils are merged
    test_tokenization_paths = [
        tokens[tokens.index(SPECIAL_TOKENS.PATH_START) :]
        for tokens in test_tokenizations
    ]

    dataset_tokenization_paths = [
        tokens[tokens.index(SPECIAL_TOKENS.PATH_START) :]
        for tokens in dataset.as_tokens(tokenizer)
    ]

    assert sorted(test_tokenization_paths) == sorted(dataset_tokenization_paths)
