from maze_dataset import MazeDataset, MazeDatasetConfig

from maze_transformer.evaluation.maze_complexity_evals import (
    MAZE_COMPLEXITY_EVALS,
    MazeComplexityEvals,
)


def test_solution_length_maze_eval():
    cfg: MazeDatasetConfig = MazeDatasetConfig(
        name="test",
        grid_n=5,
        n_mazes=3,
    )
    dataset: MazeDataset = MazeDataset.from_config(cfg).filter_by.path_length(
        min_length=2
    )

    for maze in dataset:
        solution_length: float = MazeComplexityEvals.solution_length(maze)
        assert solution_length >= 2, f"Solution length {solution_length} is less than 2"


def test_all_maze_evals_run_only():
    cfg: MazeDatasetConfig = MazeDatasetConfig(
        name="test",
        grid_n=5,
        n_mazes=3,
    )
    dataset: MazeDataset = MazeDataset.from_config(cfg).filter_by.path_length(
        min_length=2
    )

    for eval_name, eval_fn in MAZE_COMPLEXITY_EVALS.items():
        for maze in dataset:
            result: float | int = eval_fn(maze)
            assert isinstance(
                result, (float, int)
            ), f"Evaluation {eval_name} did not return a number, got {type(result) = } {result = }"
