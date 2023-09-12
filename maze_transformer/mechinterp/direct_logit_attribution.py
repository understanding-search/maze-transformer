import datetime
import json
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int

# TransformerLens imports
from transformer_lens import ActivationCache

# maze-datset stuff
from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode

# mechinterp stuff
from maze_transformer.mechinterp.logit_attrib_task import LOGIT_ATTRIB_TASKS, DLAProtocolFixed
from maze_transformer.mechinterp.logit_diff import (
    logit_diff_residual_stream,
    logits_diff_multi,
    residual_stack_to_logit_diff,
)
from maze_transformer.mechinterp.logit_lens import plot_logit_lens
from maze_transformer.mechinterp.plot_attention import plot_attention_final_token
from maze_transformer.mechinterp.plot_logits import plot_logits

# model stuff
from maze_transformer.training.config import ZanjHookedTransformer



def compute_direct_logit_attribution(
    model: ZanjHookedTransformer,
    cache: ActivationCache,
    answer_tokens: Int[torch.Tensor, "n_mazes"],
):
    # logit diff
    avg_diff, diff_direction = logit_diff_residual_stream(
        model=model,
        cache=cache,
        answer_tokens=answer_tokens,
        compare_to=None,
        directions=True,
    )

    per_head_residual, labels = cache.stack_head_results(
        layer=-1, pos_slice=-1, return_labels=True
    )
    per_head_logit_diffs = residual_stack_to_logit_diff(
        residual_stack=per_head_residual,
        cache=cache,
        logit_diff_directions=diff_direction,
    )
    per_head_logit_diffs = einops.rearrange(
        per_head_logit_diffs,
        "(layer head_index) -> layer head_index",
        layer=model.zanj_model_config.model_cfg.n_layers,
        head_index=model.zanj_model_config.model_cfg.n_heads,
    )

    return per_head_logit_diffs.to("cpu").numpy()


def plot_direct_logit_attribution(
    model: ZanjHookedTransformer,
    cache: ActivationCache,
    answer_tokens: Int[torch.Tensor, "n_mazes"],
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, Float[np.ndarray, "layer head"]]:
    data = compute_direct_logit_attribution(
        model=model,
        cache=cache,
        answer_tokens=answer_tokens,
    )

    data_extreme: float = np.max(np.abs(data))
    # colormap centeres on zero
    fig, ax = plt.subplots()
    ax.imshow(data, cmap="RdBu", vmin=-data_extreme, vmax=data_extreme)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    plt.colorbar(ax.get_images()[0], ax=ax)
    ax.set_title(f"Logit Difference from each head\n{model.zanj_model_config.name}")

    if show:
        plt.show()

    return fig, ax, data

def _output_codeblock(
    data: str|dict,
    lang: str = "",
) -> str:
    newdata: str = data

    if isinstance(data, dict):
        newdata = json.dumps(data, indent=2)

    return f"```{lang}\n{newdata}\n```"


def create_report(
    model: ZanjHookedTransformer|str|Path,
    dataset_cfg_source: MazeDatasetConfig|None,
    logit_attribution_task_name: str,
    n_examples: int = 100,
    out_path: str|Path|None = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Path:
    
    # setup
    # ======================================================================
    torch.set_grad_enabled(False)

    # model and tokenizer
    if not isinstance(model, ZanjHookedTransformer):
        model = ZanjHookedTransformer.read(model)
    tokenizer: MazeTokenizer = model.zanj_model_config.maze_tokenizer

    # dataset cfg
    if dataset_cfg_source is None:
        dataset_cfg_source = model.zanj_model_config.dataset_cfg

    # output
    if out_path is None:
        out_path = Path(f"data/dla_reports/{model.zanj_model_config.name}-{dataset_cfg_source.name}-{logit_attribution_task_name}-n{n_examples}/")

    out_path.mkdir(parents=True, exist_ok=True)
    
    fig_path: Path = out_path / "figures"
    fig_path.mkdir(parents=True, exist_ok=True)
    fig_path_md: Path = Path(f"figures")

    output_md_path: Path = out_path / "report.md"
    output_md = output_md_path.open("w")

    # write header
    output_md.write(f"""---
title: Direct Logit Attribution Report
model_name: {model.zanj_model_config.name}
dataset_cfg_name: {dataset_cfg_source.name}
logit_attribution_task_name: {logit_attribution_task_name}
n_examples: {n_examples}
time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
---

# Direct Logit Attribution Report

## Model
`{model.zanj_model_config.name}`
{_output_codeblock(model.zanj_model_config.summary(), 'json')}

## Dataset
`{dataset_cfg_source.name}`
{_output_codeblock(dataset_cfg_source.summary(), 'json')}

""")

    # task
    logit_attribution_task: DLAProtocolFixed = LOGIT_ATTRIB_TASKS[logit_attribution_task_name]

    # dataset
    dataset: MazeDataset = MazeDataset.from_config(dataset_cfg_source)
    dataset_tokens: list[list[str]] = dataset.as_tokens(tokenizer, join_tokens_individual_maze=False)

    dataset_prompts: list[list[str]]; dataset_targets: list[str]
    dataset_prompts, dataset_targets = logit_attribution_task(dataset_tokens)
    dataset_prompts_joined: list[str] = [" ".join(prompt) for prompt in dataset_prompts]
    dataset_target_ids: Float[torch.Tensor, "n_mazes"] = torch.tensor(
        tokenizer.encode(dataset_targets), dtype=torch.long
    )

    # print some info about dataset

    n_mazes: int = len(dataset)
    d_vocab: int = tokenizer.vocab_size

    output_md.write(f"""

number of mazes: {n_mazes}
vocabulary size: {d_vocab}

### First Maze

full: {_output_codeblock(' '.join(dataset_prompts[0]))}
prompt: {_output_codeblock('[...] ' + dataset_prompts_joined[0][-150:])}
target: {_output_codeblock(dataset_targets[0])}
target id: {_output_codeblock(str(dataset_target_ids[0]))}

![First maze as raster image]({fig_path_md / 'first_maze.png'})

""")
    
    plt.imsave(fig_path / "first_maze.png", dataset[0].as_pixels())

    # run model
    # ======================================================================

    logits: Float[torch.Tensor, "n_mazes seq_len d_vocab"]; cache: ActivationCache
    logits, cache = model.run_with_cache(dataset_prompts_joined, device=device)

    last_tok_logits: Float[torch.Tensor, "n_mazes d_vocab"] = logits[:, -1, :]

    output_md.write(f"""# Model Output

```
logits.shape: {logits.shape}
cache_shapes: TODO
last_tok_logits.shape: {last_tok_logits.shape}
```
""")

    plot_logits(
        last_tok_logits=last_tok_logits,
        target_idxs=dataset_target_ids,
        tokenizer=tokenizer,
        n_bins=50,
        show=False,
    )
    plt.savefig(fig_path / "last_tok_logits.png")
    output_md.write(f"![last token logits]({fig_path_md / 'last_tok_logits.png'})\n")

    predicted_tokens: list[str] = tokenizer.decode(last_tok_logits.argmax(dim=-1).tolist())
    prediction_correct: Float[torch.Tensor, "n_mazes"] = torch.tensor(
        [pred == target for pred, target in zip(predicted_tokens, dataset_targets)]
    )

    output_md.write(f"""
```
predicted_tokens.shape: {len(predicted_tokens)}
prediction_correct.shape: {prediction_correct.shape}
prediction_correct.mean(): {prediction_correct.float().mean().item()}
```
""")

    # logit diff
    logit_diff_df: pd.DataFrame = logits_diff_multi(
        model=model,
        cache=cache,
        dataset_target_ids=dataset_target_ids,
        last_tok_logits=last_tok_logits,
        noise_sigmas=np.logspace(0, 3, 100),
    )

    output_md.write(f"""
# Logit Difference
```
logit_diff_df.shape: {logit_diff_df.shape}
```

```
{logit_diff_df}
```
""")

    # scatter separately for "all" vs "random"
    fig, ax = plt.subplots()
    for compare_to in ["all", "random"]:
        df = logit_diff_df[logit_diff_df["compare_to"] == compare_to]
        ax.scatter(
            df['result_orig'], df['result_res'], 
            label=f"comparing to {compare_to}",
            marker='o',
        )
    ax.legend()
    plt.xlabel('result_orig')
    plt.ylabel('result_res')
    plt.title('Scatter Plot between result_orig and result_res')
    plt.savefig(fig_path / "logit_diff_scatter.png")
    output_md.write(f"![logit difference scatterplot comparison]({fig_path_md / 'logit_diff_scatter.png'})\n")

    # logit lens
    logitlens_figax, logitlens_results = plot_logit_lens(
        model=model,
        cache=cache,
        answer_tokens=dataset_target_ids,
        show=False,
    )
    logitlens_figax[0].savefig(fig_path / "logitlens.png")
    output_md.write(f"![logit lens results]({fig_path_md / 'logitlens.png'})\n")

    # direct logit attribution
    output_md.write(f"# Direct Logit Attribution")
    dla_fig, dla_ax, dla_data = plot_direct_logit_attribution(
        model=model,
        cache=cache,
        answer_tokens=dataset_target_ids,
        show=False,
    )
    dla_ax.set_title(f"Logit difference from each head\n{model.zanj_model_config.name}\n'{logit_attribution_task_name}' task")

    dla_fig.savefig(fig_path / "logit_attribution.png")
    output_md.write(f"![logit attribution]({fig_path_md / 'logit_attribution.png'})\n")

    # head analysis
    # let's try to plot the values of the attention heads for the top and bottom n contributing heads
    # (layer, head, value)
    top_heads: int = 5
    important_heads: list[tuple[int, int, float]] = sorted(
        [
            (i, j, dla_data[i, j])
            for i in range(dla_data.shape[0])
            for j in range(dla_data.shape[1])
        ],
        key=lambda x: abs(x[2]),
        reverse=True,
    )[:top_heads]
    # print(f"{important_heads = }")
    output_md.write(f"""
# Head Analysis
top {top_heads} heads: `{important_heads}`
""")

    # plot the attention heads
    important_heads_scores = {
        f"layer_{i}.head_{j}": (
            c,
            cache[f'blocks.{i}.attn.hook_attn_scores'][:, j, :, :].numpy(),
        )
        for i, j, c in important_heads
    }

    attn_final_tok_output: list[dict] = plot_attention_final_token(
        important_heads_scores=important_heads_scores,
        prompts=dataset_prompts,
        targets=dataset_targets,
        mazes=dataset,
        tokenizer=tokenizer,
        n_mazes=3,
        last_n_tokens=20,
        exponentiate_scores=False,
        maze_colormap_center=0.0,
        # important
        show_all=False,
        print_fmt="latex",
    )

    head_fig_path: Path = fig_path / "head_analysis"
    head_fig_path.mkdir(parents=True, exist_ok=True)
    head_fig_path_md: Path = fig_path_md / "head_analysis"

    for i, attn_final_tok in enumerate(attn_final_tok_output):
        
        head_info: dict = attn_final_tok["head_info"]
        head_lbl: str = head_info['head']

        attn_final_tok["scores"][0].savefig(head_fig_path / f"scores-{head_lbl}.png")
        attn_final_tok["attn_maze"][0].savefig(head_fig_path / f"attn_maze-{head_lbl}.png")
        
        output_md.write(f"""
## Head {head_lbl}
head info: `{head_info}`

{attn_final_tok['colored_tokens']}

![scores of attention head over tokens]({head_fig_path_md / f'scores-{head_lbl}.png'})
![scores of attention head over maze]({head_fig_path_md / f'attn_maze-{head_lbl}.png'})
""")

    # cleaning up
    output_md.flush()
    output_md.close()






