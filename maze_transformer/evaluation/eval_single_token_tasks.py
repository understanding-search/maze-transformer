# Generic

# Numerical Computing
from typing import NamedTuple

import matplotlib.pyplot as plt
import torch

# import torch.nn.functional as F
from jaxtyping import Bool, Float

# Our Code
# dataset stuff
from maze_dataset import MazeDataset
from maze_dataset.tokenization import MazeTokenizer
from muutils.json_serialize import SerializableDataclass, serializable_dataclass

# TransformerLens imports
from transformer_lens import ActivationCache

# mechinterp stuff
from maze_transformer.mechinterp.logit_attrib_task import (
    LOGIT_ATTRIB_TASKS,
    DLAProtocolFixed,
)

# model stuff
from maze_transformer.training.config import ZanjHookedTransformer

TaskPrompt = NamedTuple(
    "TaskPrompt",
    [
        ("prompts", list[str]),
        ("targets", list[str]),
    ],
)


@serializable_dataclass(kw_only=True)
class TaskEvalResult(SerializableDataclass):
    logits: Float[torch.Tensor, "samples seq_len d_vocab"]
    cache: ActivationCache | None
    last_tok_logits: Float[torch.Tensor, "samples d_vocab"]
    predicted_tokens: list[str]
    predicted_correct: Bool[torch.Tensor, "samples"]


def get_task_prompts_targets(
    dataset: MazeDataset,
    maze_tokenizer: MazeTokenizer,
    tasks: dict[str, DLAProtocolFixed] = LOGIT_ATTRIB_TASKS,
) -> dict[str, TaskPrompt]:
    dataset_tokens: list[list[str]] = dataset.as_tokens(
        maze_tokenizer,
        join_tokens_individual_maze=False,
    )

    return {task_name: task(dataset_tokens) for task_name, task in tasks.items()}


def eval_model_task(
    model: ZanjHookedTransformer,
    task: TaskPrompt,
    do_cache: bool = False,
) -> TaskEvalResult:
    maze_tokenizer: MazeTokenizer = model.tokenizer._maze_tokenizer

    prompts_joined: list[str] = [" ".join(prompt) for prompt in task.prompts]

    if do_cache:
        logits, cache = model.run_with_cache(prompts_joined)
    else:
        logits = model(prompts_joined)
        cache = None

    predicted_tokens = maze_tokenizer.decode(logits[:, -1, :].argmax(dim=-1).tolist())

    return TaskEvalResult(
        logits=logits,
        cache=cache,
        last_tok_logits=logits[:, -1, :].cpu(),
        predicted_tokens=predicted_tokens,
        predicted_correct=torch.tensor(
            [pred == target for pred, target in zip(predicted_tokens, task.targets)]
        ),
    )


def eval_model_across_tasks(
    model: ZanjHookedTransformer,
    task_prompts: dict[str, TaskPrompt],
    do_cache: bool = False,
) -> dict[str, TaskEvalResult]:
    return {
        task_name: eval_model_task(model, task, do_cache=do_cache)
        for task_name, task in task_prompts.items()
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
