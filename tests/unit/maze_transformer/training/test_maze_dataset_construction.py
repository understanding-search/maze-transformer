import pytest
from maze_dataset import SPECIAL_TOKENS, MazeDataset, MazeDatasetConfig
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode


@pytest.mark.parametrize(
    "tok_mode",
    [
        pytest.param(TokenizationMode.AOTP_UT_rasterized, id="rasterized"),
        pytest.param(TokenizationMode.AOTP_UT_uniform, id="uniform"),
    ],
)
def test_dataset_construction(tok_mode: TokenizationMode):
    config: MazeDatasetConfig = MazeDatasetConfig(
        name="test",
        grid_n=2,
        n_mazes=3,
    )
    dataset: MazeDataset = MazeDataset.from_config(cfg=config)
    tok: MazeTokenizer = MazeTokenizer(tokenization_mode=tok_mode, max_grid_size=None)

    # check the tokenization
    test_tokenizations: list[list[str]] = dataset.as_tokens(tok)

    # the adj_list always gets shuffled, so easier to check the paths
    # this will be much simpler once token utils are merged
    test_tokenization_paths = [
        tokens[tokens.index(SPECIAL_TOKENS.PATH_START) :]
        for tokens in test_tokenizations
    ]

    dataset_tokenization_paths = [
        tokens[tokens.index(SPECIAL_TOKENS.PATH_START) :]
        for tokens in dataset.as_tokens(tok)
    ]

    assert sorted(test_tokenization_paths) == sorted(dataset_tokenization_paths)
