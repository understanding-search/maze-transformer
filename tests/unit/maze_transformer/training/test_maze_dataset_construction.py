from maze_dataset import MazeDataset, MazeDatasetConfig, SPECIAL_TOKENS


def test_dataset_construction():
    config: MazeDatasetConfig = MazeDatasetConfig(
        name="test",
        grid_n=2,
        n_mazes=3,
    )
    dataset: MazeDataset = MazeDataset.from_config(cfg=config)

    # check the tokenization
    test_tokenizations: list[list[str]] = dataset.as_tokens()

    # the adj_list always gets shuffled, so easier to check the paths
    # this will be much simpler once token utils are merged
    test_tokenization_paths = [
        tokens[tokens.index(SPECIAL_TOKENS["path_start"]) :]
        for tokens in test_tokenizations
    ]

    dataset_tokenization_paths = [
        tokens[tokens.index(SPECIAL_TOKENS["path_start"]) :]
        for tokens in dataset.as_tokens()
    ]

    assert sorted(test_tokenization_paths) == sorted(dataset_tokenization_paths)
