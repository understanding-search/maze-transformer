from maze_transformer.generation.constants import SPECIAL_TOKENS
from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.training.maze_dataset import MazeDataset, MazeDatasetConfig
from maze_transformer.training.tokenizer import maze_to_tokens


def test_dataset_construction():
    n_mazes = 3
    grid_n = 2
    config = MazeDatasetConfig(
        name="test",
        grid_n=grid_n,
        n_mazes=n_mazes,
    )
    solved_mazes = []
    for _ in range(n_mazes):
        solved_maze = LatticeMazeGenerators.gen_dfs_with_solution((grid_n, grid_n))
        solved_mazes.append(solved_maze)

    dataset = MazeDataset(cfg=config, mazes_objs=solved_mazes)

    # check the tokenization

    test_tokenizations = [
        maze_to_tokens(sm, node_token_map=config.node_token_map) for sm in solved_mazes
    ]

    # the adj_list always gets shuffled, so easier to check the paths
    # this will be much simpler once token utils are merged
    test_tokenization_paths = [
        tokens[tokens.index(SPECIAL_TOKENS["path_start"]) :]
        for tokens in test_tokenizations
    ]

    dataset_tokenization_paths = [
        tokens[tokens.index(SPECIAL_TOKENS["path_start"]) :]
        for tokens in dataset.mazes_tokens
    ]

    assert sorted(test_tokenization_paths) == sorted(dataset_tokenization_paths)
