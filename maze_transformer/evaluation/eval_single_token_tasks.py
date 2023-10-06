# Generic

# Numerical Computing
import matplotlib.pyplot as plt
import torch

# import torch.nn.functional as F
from jaxtyping import Bool, Float

# Our Code
# dataset stuff
from maze_dataset import MazeDataset
from maze_dataset.tokenization import MazeTokenizer
from muutils.json_serialize import SerializableDataclass, serializable_dataclass

# mechinterp stuff
from maze_transformer.mechinterp.logit_attrib_task import (
    LOGIT_ATTRIB_TASKS,
    DLAProtocolFixed,
)

# model stuff
from maze_transformer.training.config import ZanjHookedTransformer

# TransformerLens imports


TaskPrompts = dict[
    str,  # task key
    tuple[
        list[list[str]],  # prompts
        list[str],  # targets
    ],
]


@serializable_dataclass(kw_only=True)
class TaskEvalResult(SerializableDataclass):
    logits: Float[torch.Tensor, "samples seq_len d_vocab"]
    last_tok_logits: Float[torch.Tensor, "samples d_vocab"]
    predicted_tokens: list[str]
    predicted_correct: Bool[torch.Tensor, "samples"]


def get_task_prompts_targets(
    dataset: MazeDataset,
    maze_tokenizer: MazeTokenizer,
    tasks: dict[str, DLAProtocolFixed] = LOGIT_ATTRIB_TASKS,
) -> TaskPrompts:
    dataset_tokens: list[list[str]] = dataset.as_tokens(
        maze_tokenizer,
        join_tokens_individual_maze=False,
    )

    return {task_name: task(dataset_tokens) for task_name, task in tasks.items()}


def eval_model_across_tasks(
    model: ZanjHookedTransformer,
    task_prompts: TaskPrompts,
) -> dict[str, TaskEvalResult]:
    maze_tokenizer: MazeTokenizer = model.config.maze_tokenizer

    task_logits: dict[str, Float[torch.Tensor, "n_mazes seq_len d_vocab"]] = dict()
    last_tok_logits: dict[str, Float[torch.Tensor, "n_mazes d_vocab"]] = dict()
    predicted_tokens: dict[str, list[str]] = dict()
    predictions_correct: dict[str, Bool[torch.Tensor, "n_mazes"]] = dict()

    for task_name, (prompts, targets) in task_prompts.items():
        print(f"running task {task_name}")
        prompts_joined: list[str] = [" ".join(prompt) for prompt in prompts]
        logits = model(prompts_joined)
        task_logits[task_name] = logits
        last_tok_logits[task_name] = logits[:, -1, :].cpu()
        predicted_tokens[task_name] = maze_tokenizer.decode(
            last_tok_logits[task_name].argmax(dim=-1).tolist()
        )
        predictions_correct[task_name] = torch.tensor(
            [
                pred == target
                for pred, target in zip(predicted_tokens[task_name], targets)
            ]
        )

    return {
        task_name: TaskEvalResult(
            logits=task_logits[task_name],
            last_tok_logits=last_tok_logits[task_name],
            predicted_tokens=predicted_tokens[task_name],
            predicted_correct=predictions_correct[task_name],
        )
        for task_name in task_prompts.keys()
    }


def plot_task_accuracy(
    task_results: dict[str, TaskEvalResult],
    vline_at: float | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    acc_means = {
        key: res.predicted_correct.float().mean().item()
        for key, res in task_results.items()
    }
    print(f"{acc_means = }")
    fig, ax = plt.subplots()
    # horizontal bars
    ax.barh(
        y=list(acc_means.keys()),
        width=list(acc_means.values()),
    )
    # add percentage labels
    for i, (task_name, acc) in enumerate(acc_means.items()):
        ax.text(
            x=0.1,
            y=i,
            s=f"{acc * 100:.1f}%",
            verticalalignment="center",
        )
    # vertical line (usually for 100% accuracy)
    if vline_at is not None:
        ax.axvline(x=vline_at, color="black", linestyle="--")

    return fig, ax
