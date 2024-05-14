# Generic
from pathlib import Path

# Plotting
import matplotlib.pyplot as plt

# dataset
from maze_dataset import MazeDataset
from maze_dataset.plotting import MazePlot, PathFormat

# Utilities
from muutils.mlutils import get_checkpoint_paths_for_run
from muutils.statcounter import StatCounter

# maze-transformer
from maze_transformer.evaluation.eval_model import evaluate_model, predict_maze_paths
from maze_transformer.training.config import ZanjHookedTransformer


def plot_predicted_paths(
    model: ZanjHookedTransformer,
    dataset: MazeDataset,
    n_mazes: int | None = None,
    max_new_tokens: int = 8,
    show: bool = True,
    remove_labels: bool = True,
    row_length: int | None = None,
    figsize_scale: int = 10,
    predicted_path_fmt: PathFormat | None = None,
    print_generations: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    if n_mazes is None:
        n_mazes = len(dataset)

    dataset_tokens = dataset.as_tokens(model.tokenizer._maze_tokenizer)[:n_mazes]

    # predict
    predictions: list[list[str | tuple[int, int]]] = predict_maze_paths(
        tokens_batch=dataset_tokens,
        data_cfg=dataset.cfg,
        model=model,
        max_new_tokens=max_new_tokens,
        when_noncoord="include",
        smart_max_new_tokens=True,
        batch_size=32,
    )

    if print_generations:
        for x in predictions:
            print(" ".join([str(t) for t in x]))

    predictions_filtered: list[list[tuple[int, int]]] = [
        [token for token in path if not isinstance(token, str)] for path in predictions
    ]

    # fig, axs = plt.subplots(1, n_mazes, figsize=(10, 10 * n_mazes))
    if row_length is None:
        row_length = n_mazes
    n_rows = n_mazes // row_length
    if n_mazes % row_length != 0:
        n_rows += 1

    fig, axs = plt.subplots(
        n_rows, row_length, figsize=(figsize_scale * row_length, figsize_scale * n_rows)
    )
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    # plot
    for i, maze in enumerate(dataset.mazes[:n_mazes]):
        ax_idx = i // row_length, i % row_length
        ax = axs[ax_idx] if n_rows > 1 else axs[i]
        mp: MazePlot = MazePlot(maze).add_predicted_path(
            predictions_filtered[i],
            path_fmt=predicted_path_fmt,
        )
        mp.plot(fig_ax=(fig, ax))

        if remove_labels:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")

    if show:
        plt.show()

    return fig, axs


def eval_model_at_checkpoints(
    model_path: Path,
    dataset: MazeDataset,
    max_checkpoints: int = 50,
) -> dict[str, dict[int, StatCounter]]:
    """runs evaluate_model on various checkpoints of a model

    returned dict maps eval name to a dict of checkpoint index to statcounter
    """

    model_checkpoints: list[tuple[int, Path]] = get_checkpoint_paths_for_run(
        model_path.parent, "zanj"
    )
    model_checkpoints.sort(key=lambda x: x[0])
    print(
        f"Found {len(model_checkpoints)} checkpoints, min_index={model_checkpoints[0][0]}, max_index={model_checkpoints[-1][0]}"
    )

    # filter to every nth checkpoint
    num_checkpoints: int = len(model_checkpoints)
    take_every_nth: int = max(1, num_checkpoints // max_checkpoints)
    model_checkpoints: list[tuple[int, Path]] = model_checkpoints[::take_every_nth]
    print(
        f"will evaluate {len(model_checkpoints)} checkpoints: {[(i,p.as_posix()) for i,p in model_checkpoints]}"
    )

    pathdist_scores_idx: dict[int, dict[str, StatCounter]] = dict()

    for idx, checkpoint_path in model_checkpoints:
        print(f"# Evaluating checkpoint {idx} at {checkpoint_path}")
        model_at_index = ZanjHookedTransformer.read(checkpoint_path)
        pathdist_scores_idx[idx] = evaluate_model(
            model=model_at_index,
            dataset=dataset,
        )

    return {
        name: {idx: scores[name] for idx, scores in pathdist_scores_idx.items()}
        for name in pathdist_scores_idx[list(pathdist_scores_idx.keys())[0]].keys()
    }


def plot_pathdist_scores(
    data: dict[str, dict[int, StatCounter]],
    colors: dict[str, str] | None = None,
    percentile_bounds: tuple[float, float] = (0.4, 0.6),
):
    """plots pathdist scores over checkpoints

    expects a dict mapping eval name to a dict of checkpoint index to statcounter
    """

    if colors is None:
        colors = {func_name: f"C{i}" for i, func_name in enumerate(data.keys())}

    fig, ax = plt.subplots(len(data), 1, figsize=(8, 4 * len(data)))
    fig.subplots_adjust(hspace=0.5)

    for i, (name, scores_indexed) in enumerate(data.items()):
        x = list(scores_indexed.keys())
        y = [scores_indexed[i].median() for i in x]
        ax[i].plot(x, y, label=name, color=colors[name])
        # plot shaded error bars
        y_ub = [scores_indexed[i].percentile(percentile_bounds[1]) for i in x]
        y_lb = [scores_indexed[i].percentile(percentile_bounds[0]) for i in x]
        ax[i].fill_between(
            x,
            y_lb,
            y_ub,
            alpha=0.5,
            edgecolor=colors[name],
            facecolor=colors[name],
        )

        ax[i].set_title(f"{name}, {percentile_bounds = }")
        ax[i].set_xlabel("Checkpoint")
        ax[i].set_ylabel("score")

    plt.show()
