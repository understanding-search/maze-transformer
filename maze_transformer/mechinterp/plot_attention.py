# Generic
import typing
from collections import defaultdict

# plotting
import matplotlib.pyplot as plt

# Numerical Computing
import numpy as np
import torch

# Transformers
from circuitsvis.attention import attention_heads
from circuitsvis.tokens import colored_tokens_multi
from jaxtyping import Float
from maze_dataset import CoordTup, MazeDataset, MazeDatasetConfig, SolvedMaze
from maze_dataset.plotting import MazePlot
from maze_dataset.plotting.print_tokens import color_tokens_cmap
from maze_dataset.plotting.plot_tokens import plot_colored_text
from maze_dataset.tokenization import MazeTokenizer
from maze_dataset.tokenization.token_utils import coord_str_to_tuple_noneable

# Utilities
from muutils.json_serialize import SerializableDataclass, serializable_dataclass

from maze_transformer.evaluation.eval_model import predict_maze_paths
from maze_transformer.tokenizer import SPECIAL_TOKENS
from maze_transformer.training.config import ZanjHookedTransformer


@serializable_dataclass
class ProcessedMazeAttention(SerializableDataclass):
    input_maze: SolvedMaze
    tokens: list[str]
    tokens_context: list[str]
    logits: Float[torch.Tensor, "n_vocab"]
    n_layers: int
    n_heads: int
    attention_dict: dict[str, Float[torch.Tensor, "1 n_heads n_positions n_positions"]]
    attention_tensored: Float[torch.Tensor, "n_layers n_heads n_tokens n_tokens"]
    attention_names: list[str]  # names for attention_tensored

    @classmethod
    def from_model_and_dataset(
        cls,
        model: ZanjHookedTransformer,
        dataset: MazeDataset,
        n_mazes: int = 1,
        context_maze_only: bool = True,
        context_maze_fn: typing.Callable[[list[str]], list[str]] | None = None,
    ) -> list["ProcessedMazeAttention"]:
        outputs: list[ProcessedMazeAttention] = list()

        for i in range(n_mazes):
            # get the maze from the dataset and process into tokens
            solved_maze: SolvedMaze = dataset[i]
            tokens: list[str] = solved_maze.as_tokens(
                model.zanj_model_config.maze_tokenizer
            )
            tokens_context: list[str]

            if context_maze_only:
                assert context_maze_fn is None
                path_start_index: int = tokens.index(SPECIAL_TOKENS.PATH_END)
                tokens_context = tokens[: path_start_index + 1]
            else:
                assert context_maze_fn is not None
                tokens_context = context_maze_fn(tokens)

            # get the model's prediction and attention data
            with torch.no_grad():
                # we have to join here, since otherwise run_with_cache assumes each token is a separate batch
                logits, cache = model.run_with_cache(" ".join(tokens_context))

            # filter and append to outputs
            attention_data: dict[str, torch.Tensor] = {
                k: w for k, w in cache.items() if "hook_pattern" in k
            }

            assert model.zanj_model_config.model_cfg.n_layers == len(attention_data)
            example_attention_data: Float[
                torch.Tensor, "1 n_heads n_positions n_positions"
            ] = attention_data[list(attention_data.keys())[0]]
            assert (
                model.zanj_model_config.model_cfg.n_heads
                == example_attention_data.shape[1]
            )
            n_tokens: int = example_attention_data.shape[2]

            attention_tensored: Float[
                torch.Tensor, "n_layers_heads n_tokens n_tokens"
            ] = torch.concatenate(
                [w for k, w in attention_data.items()],
                dim=0,
            ).reshape(
                -1, n_tokens, n_tokens
            )

            outputs.append(
                ProcessedMazeAttention(
                    input_maze=solved_maze,
                    tokens=tokens,
                    tokens_context=tokens_context,
                    logits=logits,
                    n_layers=model.zanj_model_config.model_cfg.n_layers,
                    n_heads=model.zanj_model_config.model_cfg.n_heads,
                    attention_dict=attention_data,
                    attention_tensored=attention_tensored,
                    attention_names=[
                        f"Layer {i} Head {j}"
                        for i in range(model.zanj_model_config.model_cfg.n_layers)
                        for j in range(model.zanj_model_config.model_cfg.n_heads)
                    ],
                )
            )

        return outputs

    def summary(self) -> dict:
        return {
            "tokens": " ".join(self.tokens),
            "tokens_context": " ".join(self.tokens_context),
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "logits.shape": self.logits.shape,
            "attention_tensored.shape": self.attention_tensored.shape,
            "attention_names": " ".join(self.attention_names),
            "attention_dict.keys()": " ".join(list(self.attention_dict.keys())),
        }

    def plot_attentions(self, **kwargs):
        """plot using circuitsvis attention_heads. kwargs passed on"""
        return attention_heads(
            self.attention_tensored,
            self.tokens_context,
            self.attention_names,
            **kwargs,
        )

    def plot_colored_tokens_multi(self, from_token: int = 0, **kwargs):
        """plot using circuitsvis colored_tokens_multi. kwargs passed on"""
        attentions_from_token: Float[
            torch.Tensor, "n_tokens-from_token n_layers_heads"
        ] = torch.sum(
            self.attention_tensored[:, from_token + 1 :, from_token + 1 :],
            dim=2,  # TODO: is this the correct dimension?
        ).T
        return colored_tokens_multi(
            self.tokens_context[from_token:],
            attentions_from_token,
            self.attention_names,
            **kwargs,
        )

    def plot_attentions_on_maze(
        self,
        predict_path_len: int | None = None,
        model: ZanjHookedTransformer | None = None,
        dataset_cfg: MazeDatasetConfig | None = None,
        color_map: str = "Blues",
        subplot_kwargs: dict | None = None,
    ):
        """plot the attention weights on the maze itself, for each head

        if predict_path_len is not None, then the path will be predicted and plotted as well
        this requires a model and dataset_cfg to be passed in
        if it is None, no path will be predicted or plotted

        subplot_kwargs are passed to the plt.subplots call
        """
        if subplot_kwargs is None:
            subplot_kwargs = dict()
        # add predicted path
        if predict_path_len is not None:
            # predict
            print(self.tokens_context)
            prediction: list[CoordTup] = predict_maze_paths(
                tokens_batch=[
                    self.tokens_context
                ],  # need to wrap in a list since a batch is expected
                data_cfg=dataset_cfg,
                model=model,
                max_new_tokens=predict_path_len,
            )[0]

        # initialize MazePlots for each attention head
        mazeplots: list[list[MazePlot]] = [
            [
                (
                    MazePlot(self.input_maze)
                    if predict_path_len is None
                    else MazePlot(self.input_maze).add_predicted_path(prediction)
                )
                for j in range(self.n_heads)
            ]
            for i in range(self.n_layers)
        ]

        for idx_attn, (name, attn) in enumerate(
            zip(self.attention_names, self.attention_tensored)
        ):
            # --------------------------
            # empty node values
            node_values: Float[np.ndarray, "grid_n grid_n"] = np.zeros(
                self.input_maze.grid_shape
            )
            # get node values for each token
            for idx_token, token in enumerate(self.tokens_context):
                coord: CoordTup | None = coord_str_to_tuple_noneable(token)
                if coord is not None:
                    node_values[coord[0], coord[1]] += np.sum(
                        attn[idx_token].cpu().numpy()
                    )

                # update MazePlot objects
                mazeplots[idx_attn // self.n_heads][
                    idx_attn % self.n_heads
                ].add_node_values(
                    node_values=node_values,
                    color_map=color_map,
                )
            # --------------------------

        # create a shared figure
        fig, axs = plt.subplots(
            self.n_layers, self.n_heads, figsize=(20, 20), **subplot_kwargs
        )
        # plot on the shared figure and add titles
        for i in range(self.n_layers):
            for j in range(self.n_heads):
                mazeplots[i][j].plot(
                    title=f"Layer {i} Head {j}",
                    fig_ax=(fig, axs[i, j]),
                )

        return fig, axs


def mazeplot_attention(
    maze: SolvedMaze,
    tokens_context: str,
    attention: Float[np.ndarray, "n_tokens"],
    mazeplot: MazePlot | None = None,
    color_maps: tuple[str, str] = (
        "plasma",
        "RdBu",
    ),  # all positive, positive and negative
    min_for_positive: float = 0.0,
    show_other_tokens: bool = True,
) -> tuple[MazePlot, plt.Figure, plt.Axes]:
    
    # set up color map
    if attention.min() >= min_for_positive:
        cmap = color_maps[0]
    else:
        cmap = color_maps[1]

    # storing attention
    node_values: Float[np.ndarray, "grid_n grid_n"] = np.zeros(maze.grid_shape)
    total_logits_nonpos = defaultdict(float)

    # get node values for each token
    for idx_token, token in enumerate(tokens_context):
        coord: CoordTup | None = coord_str_to_tuple_noneable(token)
        if coord is not None:
            node_values[coord[0], coord[1]] += np.sum(attention[idx_token])
        else:
            total_logits_nonpos[token] += attention[idx_token]

    # MazePlot attentions
    if mazeplot is None:
        mazeplot = MazePlot(maze)

    mazeplot.add_node_values(
        node_values=node_values,
        color_map=cmap,
    )

    # set up combined figure
    fig, (ax_maze, ax_other) = plt.subplots(
        2, 1, 
        figsize=(7, 11.5),
        height_ratios=[7, 1],
    )
    # set height ratio
    mazeplot.plot(
        title=f"{attention.min() = }\n{attention.max() = }",
        fig_ax=(fig, ax_maze),
    )

    # non-pos tokens attention
    total_logits_nonpos_processed: tuple[list[str], list[float]] = tuple(zip(*
        sorted(
            total_logits_nonpos.items(), key=lambda x: x[0]
        )
    ))

    plot_colored_text(
        total_logits_nonpos_processed[0],
        total_logits_nonpos_processed[1],
        cmap=cmap,
        ax=ax_other,
        fontsize=5,
        width_scale=0.01,
        char_min=5,
    )

    ax_other.set_title("Non-Positional Tokens Attention")

    return mazeplot, fig, (ax_maze, ax_other)


def plot_attention_final_token(
    important_heads_scores: dict[
        str,
        tuple[float, Float[np.ndarray, "n_mazes n_tokens n_tokens"]],
    ],
    prompts: list[list[str]],
    mazes: list[SolvedMaze],
    tokenizer: MazeTokenizer,
    n_mazes: int = 5,
    last_n_tokens: int = 20,
    exponentiate_scores: bool = False,
):
    for k, (c, v) in important_heads_scores.items():
        print(f"{k = }, {c = } {v.shape = }")

        # set up scores across tokens figure
        scores_fig, scores_ax = plt.subplots(n_mazes, 1)
        scores_fig.set_size_inches(30, 4 * n_mazes)
        # for each maze
        for i in range(n_mazes):
            # process tokens and attention scores
            n_tokens_prompt = len(prompts[i])
            n_tokens_view = min(n_tokens_prompt, last_n_tokens)
            v_final = v[i][-1]  # -1 for last token
            if exponentiate_scores:
                v_final = np.exp(v_final)

            # print token scores
            print(
                color_tokens_cmap(
                    prompts[i][-n_tokens_view:],
                    v_final[-n_tokens_view:],
                    fmt="terminal",
                    labels=True,
                )
            )

            # plot across tokens
            scores_ax[i].plot(
                v_final[-n_tokens_prompt:],
                "o",
            )
            scores_ax[i].grid(axis="x", which="major", color="black", alpha=0.1)
            scores_ax[i].set_xticks(range(n_tokens_prompt), prompts[i], rotation=90)

            # plot attention across maze
            mazeplot_attention(
                maze=mazes[i],
                tokens_context=prompts[i][-n_tokens_view:],
                attention=v_final[-n_tokens_view:],
            )

        plt.show()
