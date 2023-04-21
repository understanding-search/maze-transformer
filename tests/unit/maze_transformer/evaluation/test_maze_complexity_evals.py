from maze_transformer.evaluation.maze_complexity_evals import (
    MAZE_COMPLEXITY_EVALS,
    MazeComplexityEvals,
)
from maze_transformer.training.maze_dataset import MazeDataset, MazeDatasetConfig

cfg: MazeDatasetConfig = MazeDatasetConfig(
    name="test",
    grid_n=5,
    n_mazes=3,
)
dataset: MazeDataset = MazeDataset.from_config(cfg)


def test_solution_length_maze_eval():
    dataset: MazeDataset = MazeDataset.from_config(cfg).filter_by.path_length(
        min_length=2
    )

    for maze in dataset.mazes:
        solution_length: float = MazeComplexityEvals.solution_length(maze)
        assert solution_length >= 2, f"Solution length {solution_length} is less than 2"


def test_decisions_in_solution_norm():
    for maze in dataset.mazes:
        norm: float = MazeComplexityEvals.decisions_in_solution_norm(
            maze=maze, solution=maze.solution
        )
        assert norm >= 0, f"Normalized Decisions {norm} is less than 0"
        assert norm <= 1, f"Normalized Decisions {norm} is greater than 1"


def test_decisions_in_solution_score():
    for maze in dataset.mazes:
        score: float = MazeComplexityEvals.decisions_in_solution_score(
            maze=maze, solution=maze.solution
        )
        assert score >= 1, f"Decisions in solution score {score} is less than 0"
        assert score <= len(
            maze.solution
        ), f"Decisions in solution score {score} is greater than the solution length {len(maze.solution)}"


def test_adj_list_length():
    for maze in dataset.mazes:
        result: int = MazeComplexityEvals.adj_list_length(maze=maze)
        assert isinstance(
            result, int
        ), f"Evaluation {result} did not return a number, got {type(result) = } {result = }"


def test_all_maze_evals_run_only():
    for eval_name, eval_fn in MAZE_COMPLEXITY_EVALS.items():
        for maze in dataset.mazes:
            result: float | int = eval_fn(maze=maze, solution=maze.solution)
            assert isinstance(
                result, (float, int)
            ), f"Evaluation {eval_name} did not return a number, got {type(result) = } {result = }"


if __name__ == "__main__":
    test_adj_list_length()
    test_decisions_in_solution_norm()
    test_decisions_in_solution_score()
    test_solution_length_maze_eval()
    test_all_maze_evals_run_only()
    