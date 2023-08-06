import pytest
from maze_dataset import MazeDataset, MazeDatasetConfig, SolvedMaze
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode

from maze_transformer.test_helpers.stub_logger import StubLogger
from maze_transformer.training.config import GPT_CONFIGS, TRAINING_CONFIGS, ConfigHolder
from maze_transformer.training.training import get_dataloader


@pytest.mark.parametrize(
    "tok_mode",
    [
        pytest.param(TokenizationMode.AOTP_UT_rasterized, id="rasterized"),
        pytest.param(TokenizationMode.AOTP_UT_uniform, id="uniform"),
    ],
)
def test_get_dataloader(tok_mode: TokenizationMode):
    dataset_config = MazeDatasetConfig(name="test", grid_n=3, n_mazes=5)
    dataset = MazeDataset.generate(dataset_config)
    config_holder: ConfigHolder = ConfigHolder(
        dataset_cfg=dataset_config,
        model_cfg=GPT_CONFIGS["tiny-v1"],
        train_cfg=TRAINING_CONFIGS["test-v1"],
        maze_tokenizer=MazeTokenizer(tokenization_mode=tok_mode),
    )
    config_holder.train_cfg.batch_size = 5
    logger = StubLogger()
    dataloader = get_dataloader(dataset, config_holder, logger)
    dataloader_iter = iter(dataloader)

    batch1 = next(dataloader_iter)

    other_batch1 = next(iter(dataloader))
    dataloader_mazes = [
        SolvedMaze.from_tokens(tokens, config_holder.maze_tokenizer)
        for tokens in batch1
    ]

    assert len(batch1) == 5
    # The bataloader batch should contain the same mazes as the dataset
    assert all(
        any(dataloader_maze == dataset_maze for dataset_maze in dataset)
        for dataloader_maze in dataloader_mazes
    )
    assert batch1 != other_batch1  # adj_list is shuffled for every sample
