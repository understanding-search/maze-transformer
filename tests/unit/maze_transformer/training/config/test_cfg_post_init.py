from maze_dataset import MazeDatasetConfig
from maze_dataset.tokenization import MazeTokenizer

from maze_transformer.training.config import BaseGPTConfig, ConfigHolder, TrainConfig


def test_cfg_post_init():
    cfg: ConfigHolder = ConfigHolder(
        train_cfg=TrainConfig(name="test_cfg_save-train"),
        dataset_cfg=MazeDatasetConfig(name="test_cfg_save-data", grid_n=5, n_mazes=10),
        model_cfg=BaseGPTConfig(
            name="test_cfg_save-model",
            act_fn="dummy-act-fn",
            d_model=500,
            d_head=60,
            n_layers=4,
        ),
    )

    assert isinstance(cfg.maze_tokenizer, MazeTokenizer)
    assert isinstance(cfg.maze_tokenizer.max_grid_size, int)
    assert cfg.maze_tokenizer.max_grid_size == 5
    assert isinstance(cfg.maze_tokenizer.vocab_size, int)
