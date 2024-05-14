# Generic
import typing
from collections import defaultdict

# plotting
import matplotlib.pyplot as plt

# Numerical Computing
import numpy as np
import seaborn as sns
import torch

# Transformers
from circuitsvis.attention import attention_heads
from circuitsvis.tokens import colored_tokens_multi
from jaxtyping import Float
from maze_dataset import CoordTup, MazeDataset, MazeDatasetConfig, SolvedMaze
from maze_dataset.plotting import MazePlot
from maze_dataset.plotting.plot_tokens import plot_colored_text
from maze_dataset.plotting.print_tokens import color_tokens_cmap
from maze_dataset.tokenization import MazeTokenizer
from maze_dataset.tokenization.util import coord_str_to_tuple_noneable

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
    target: str,
    attention: Float[np.ndarray, "n_tokens"],
    mazeplot: MazePlot | None = None,
    cmap: str = "RdBu",
    # min_for_positive: float = 0.0,
    show_other_tokens: bool = True,
    plain_figure: bool = False,
    fig_ax: tuple[plt.Figure, plt.Axes] | None = None,
    colormap_center: None | float | typing.Literal["median", "mean"] = None,
    colormap_max: None | float = None,
    hide_colorbar: bool = False,
    cbar_height_factor: float = 0.97,
) -> tuple[MazePlot, plt.Figure, plt.Axes]:
    # storing attention
    node_values: Float[np.ndarray, "grid_n grid_n"] = np.zeros(maze.grid_shape)
    total_logits_nonpos = defaultdict(float)

    # get node values for each token
    for idx_token, token in enumerate(tokens_context):
        coord: CoordTup | None = coord_str_to_tuple_noneable(token)
        # TODO: mean/median instead of just sum?
        if coord is not None:
            node_values[coord[0], coord[1]] += np.sum(attention[idx_token])
        else:
            total_logits_nonpos[token] += attention[idx_token]

    # MazePlot attentions
    if mazeplot is None:
        mazeplot = MazePlot(maze)

    final_prompt_coord: CoordTup | None = coord_str_to_tuple_noneable(
        tokens_context[-1]
    )
    target_coord: CoordTup | None = coord_str_to_tuple_noneable(target)

    colormap_center_val: float | None
    if colormap_center is None:
        colormap_center_val = None
    elif colormap_center == "median":
        colormap_center_val = np.median(attention)
    elif colormap_center == "mean":
        colormap_center_val = np.mean(attention)
    else:
        colormap_center_val = colormap_center

    mazeplot.add_node_values(
        node_values=node_values,
        color_map=cmap,
        target_token_coord=target_coord,
        preceeding_tokens_coords=(
            [final_prompt_coord] if final_prompt_coord is not None else None
        ),
        colormap_center=colormap_center_val,
        colormap_max=colormap_max,
        hide_colorbar=hide_colorbar,
    )

    # set up combined figure
    if fig_ax is None:
        fig_ax = plt.subplots(
            # adding a second row for non-pos tokens
            1 + int(show_other_tokens),
            1,
            figsize=(7, 7),
            **(dict(height_ratios=[7, 1]) if show_other_tokens else dict()),
        )

    if show_other_tokens:
        fig, (ax_maze, ax_other) = fig_ax
    else:
        fig, ax_maze = fig_ax

    # add min and max in title
    mp_title: str | None = (
        None if plain_figure else f"{attention.min() = }\n{attention.max() = }"
    )
    mazeplot.plot(
        title=mp_title,
        fig_ax=(fig, ax_maze),
    )

    # adjust the height of the colorbar
    # TODO: move this to MazePlot
    if mazeplot.cbar_ax is not None:
        pos = mazeplot.cbar_ax.get_position()
        new_height: float = pos.height * cbar_height_factor
        new_y0: float = pos.y0 + (pos.height - new_height) / 2
        mazeplot.cbar_ax.set_position([pos.x0, new_y0, pos.width, new_height])
        # add a title to the colorbar, vertically and to the side
        mazeplot.cbar_ax.text(
            5.0,
            0.5,
            "Attention",
            rotation=90,
            verticalalignment="center",
            transform=mazeplot.cbar_ax.transAxes,
        )

    if plain_figure:
        # remove axis ticks
        ax_maze.set_xticks([])
        ax_maze.set_yticks([])
        ax_maze.set_xlabel(None)
        ax_maze.set_ylabel(None)
        ax_maze.set_xticklabels([])
        ax_maze.set_yticklabels([])

    # non-pos tokens attention
    if show_other_tokens:
        total_logits_nonpos_processed: tuple[list[str], list[float]] = tuple(
            zip(*sorted(total_logits_nonpos.items(), key=lambda x: x[0]))
        )

        if len(total_logits_nonpos_processed) == 2:
            plot_colored_text(
                total_logits_nonpos_processed[0],
                total_logits_nonpos_processed[1],
                cmap=cmap,
                ax=ax_other,
                fontsize=5,
                width_scale=0.01,
                char_min=5,
            )
        else:
            print(f"No non-pos tokens found!\n{total_logits_nonpos_processed = }")

        ax_other.set_title("Non-Positional Tokens Attention")

    if show_other_tokens:
        return mazeplot, fig, (ax_maze, ax_other)
    else:
        return mazeplot, fig, ax_maze


def plot_attn_dist_correlation(
    tokens_context: list[list[str]],
    tokens_dist_to: list[str],  # either current or target token for each maze
    tokenizer: MazeTokenizer,
    attention: Float[np.ndarray, "n_mazes n_tokens"],
    ax: plt.Axes | None = None,
    respect_topology: bool = False,  # manhattan distance if False
    xlim: int = 10,
) -> plt.Axes:
    # print(attention.shape)
    assert len(tokens_context) == attention.shape[0]
    # decode the tokens to coordinates
    coords_context: list[list[tuple[int, int]]] = [
        tokenizer.strings_to_coords(tokens, when_noncoord="include")
        for tokens in tokens_context
    ]
    coords_dist_to: list[tuple[int, int]] = tokenizer.strings_to_coords(
        tokens_dist_to,
        when_noncoord="include",
    )

    attention_lst: list[Float[np.ndarray, "n_tokens"]] = [
        a[-len(c) :] for i, (c, a) in enumerate(zip(coords_context, attention))
    ]
    # compute the distances
    distances: list[Float[np.ndarray, "n_tokens"]] = list()
    if respect_topology:
        # convert context to maze, compute shortest path
        mazes: list[SolvedMaze] = [
            SolvedMaze.from_tokens(tokens, maze_tokenizer=tokenizer)
            for tokens in tokens_context
        ]
        distances = [
            np.array(
                [
                    (
                        maze.find_shortest_path(coords_dist_to[idx], c).shape[0] - 1
                        if not isinstance(c, str)
                        else np.inf
                    )
                    for c in coords_context[idx]
                ]
            )
            for idx, maze in enumerate(mazes)
        ]
    else:
        distances = [
            np.array(
                [
                    (
                        np.sum(np.abs(np.array(coords_dist_to[idx]) - np.array(c)))
                        if not isinstance(c, str)
                        else np.inf
                    )
                    for c in coords_context[idx]
                ]
            )
            for idx in range(len(coords_context))
        ]

    # plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))

    dists_plot = np.concatenate(distances, axis=0)
    attn_plot = np.concatenate(attention_lst, axis=0)
    # remove elements where dists_plot is inf
    nan_mask = ~np.isinf(dists_plot)
    dists_plot = dists_plot[nan_mask]
    attn_plot = attn_plot[nan_mask]

    # print(f"{dists_plot.shape = }, {attn_plot.shape = }")
    # print(f"{dists_plot.dtype = }, {attn_plot.dtype = }")

    sns.violinplot(
        x=dists_plot,
        y=attn_plot,
        ax=ax,
        color=sns.color_palette()[0],
        # bw_adjust=10.0,
        # bw_method="silverman",
        bw=2.0,
        scale="count",
        cut=0,
        # jitter=True,
        # alpha=0.5,
    )

    # rotate xticks
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlim(-0.5, xlim + 0.5)

    ax.set_xlabel(f"manhattan distance, {respect_topology = }")
    ax.set_ylabel("attention")

    return ax, distances, attention_lst


def plot_attention_final_token(
    important_heads_scores: dict[
        str,
        tuple[float, Float[np.ndarray, "n_mazes n_tokens n_tokens"]],
    ],
    prompts: list[list[str]],
    targets: list[str],
    mazes: list[SolvedMaze],
    tokenizer: MazeTokenizer,
    n_mazes: int = 5,
    last_n_tokens: int = 20,
    # exponentiate_scores: bool = False,
    softmax_attention: bool = True,
    plot_attn_dist_corr: bool = True,
    attn_dist_to: typing.Literal["current", "target"] = "current",
    plot_colored_tokens: bool = True,
    plot_scores: bool = True,
    plot_attn_maze: bool = True,
    maze_colormap_center: None | float | typing.Literal["median", "mean"] = None,
    max_colormap_max: float = 0.5,
    mazeplot_simplified: bool = True,
    mazeplot_attn_cmap: str = "RdBu",
    show_all: bool = True,
    print_fmt: str = "terminal",
    plotshow_func: typing.Callable[[str], None] | None = None,
) -> list[dict]:
    # str, # head info
    # str|None, # colored tokens text
    # tuple[plt.Figure, plt.Axes]|None, # scores plot
    # tuple[plt.Figure, plt.Axes]|None, # attn maze plot
    output: list[dict[str, str | None | tuple[plt.Figure, plt.Axes]]] = list()

    for k, (c, attn_presoftmax) in important_heads_scores.items():
        head_info: str = f"head: {k}, score: {c = }, {attn_presoftmax.shape = }"
        if show_all:
            print("-" * 80)
            print(head_info)

        head_output: dict[str, str | None | tuple[plt.Figure, plt.Axes]] = dict(
            head_info_str=head_info,
            head_info=dict(
                head=k,
                score=c,
                shape=attn_presoftmax.shape,
            ),
        )

        # process attention, getting the last token attention and maybe softmaxing
        if softmax_attention:
            attn = torch.softmax(
                torch.tensor(attn_presoftmax[:, -1]),
                dim=-1,
            ).numpy()
        else:
            attn = attn_presoftmax[:, -1]

        # set up attn dist corr figure
        if plot_attn_dist_corr:
            for respect_topology in [False, True]:
                ax, distcorr_dist, distcorr_attn = plot_attn_dist_correlation(
                    tokens_context=prompts,
                    tokens_dist_to=(
                        [x[-1] for x in prompts]
                        if attn_dist_to == "current"
                        else targets
                    ),
                    tokenizer=tokenizer,
                    # apply softmax to attention
                    attention=attn,
                    respect_topology=respect_topology,
                )
                ax.set_title(k)
                if plotshow_func is not None:
                    plotshow_func(
                        f"attn_dist_corr-{'topology' if respect_topology else 'lattice'}-{k}"
                    )
                else:
                    plt.show()

            head_output["attn_dist_corr"] = (distcorr_dist, distcorr_attn)

        # set up scores across tokens figure
        if plot_scores:
            scores_fig, scores_ax = plt.subplots(n_mazes, 1)
            scores_fig.set_size_inches(30, 4 * n_mazes)

        # set up attention across maze figure
        if plot_attn_maze:
            mazes_fig, mazes_ax = plt.subplots(
                1 if mazeplot_simplified else 2,
                n_mazes,
                figsize=(7 * n_mazes, 7),
                height_ratios=[7, 1] if not mazeplot_simplified else None,
            )

        # for each maze
        for i in range(n_mazes):
            # process tokens and attention scores
            n_tokens_prompt = len(prompts[i])
            n_tokens_view = min(n_tokens_prompt, last_n_tokens)
            v_final = attn[i]
            # print(f"{attn.shape}, {v_final.shape = }")

            # print token scores
            if plot_colored_tokens:
                color_tokens_text: str = color_tokens_cmap(
                    prompts[i][-n_tokens_view:],
                    v_final[-n_tokens_view:],
                    fmt=print_fmt,
                    labels=(print_fmt == "terminal"),
                )
                if show_all:
                    print(color_tokens_text)

                head_output["colored_tokens"] = color_tokens_text

            # plot across tokens
            if plot_scores:
                scores_ax[i].plot(
                    v_final[-n_tokens_prompt:],
                    "o",
                )
                scores_ax[i].grid(axis="x", which="major", color="black", alpha=0.1)
                scores_ax[i].set_xticks(range(n_tokens_prompt), prompts[i], rotation=90)

                head_output["scores"] = (scores_fig, scores_ax)

            # plot attention across maze
            if plot_attn_maze:
                mazeplot, fig, ax = mazeplot_attention(
                    maze=mazes[i],
                    tokens_context=prompts[i][-n_tokens_prompt:],
                    target=targets[i],
                    attention=v_final[-n_tokens_prompt:],
                    fig_ax=(
                        mazes_fig,
                        mazes_ax[i] if mazeplot_simplified else mazes_ax[:, i],
                    ),
                    colormap_center=maze_colormap_center,
                    cmap=mazeplot_attn_cmap,
                    plain_figure=mazeplot_simplified,
                    show_other_tokens=not mazeplot_simplified,
                    colormap_max=max_colormap_max,
                    hide_colorbar=mazeplot_simplified,
                )

                head_output["attn_maze"] = (mazes_fig, mazes_ax)

        # put a shared colorbar on mazes_fig
        if plot_attn_maze:
            mazes_fig.colorbar(
                ax.get_images()[0],
                ax=mazes_ax,
                location="right",
                shrink=0.7,
                pad=0.01,
            )

        if show_all:
            plt.show()
        else:
            output.append(head_output)

    return output


def plot_attention_anim(
    cache: "ActivationCache",
    maze_id: int,
    mazes: list[SolvedMaze],
    mazes_tokens: list[list[str]],
    head_id: tuple[int, int],
    end_offset: int = -2,
    fps: int = 2,
    figsize: tuple[float, float] = (7, 7),
):
    """plot an animation of a head's attention over the maze

    # Parameters:
     - `cache : ActivationCache`
        cache of activations from the model
     - `maze : SolvedMaze`
        maze to plot
     - `maze_tokens : list[str]`
        tokens fed to the model
     - `head_id : tuple[int, int]`
        (head_layer, head_index) of the head we want to plot
     - `end_offset : int`
        offset from the end of the stream, -2 for not including `<PATH_END>` token
       (defaults to `-2`)
    """
    from celluloid import Camera

    maze: SolvedMaze = mazes[maze_id]
    maze_tokens: list[str] = mazes_tokens[maze_id]

    head_layer, head_index = head_id

    head_cache: Float[np.ndarray, "n_mazes seq_len seq_len"] = (
        cache[f"blocks.{head_layer}.attn.hook_attn_scores"][:, head_index, :, :]
        .cpu()
        .numpy()
    )

    maze_tokens: list[str] = maze_tokens

    path_idx_start: int = maze_tokens.index(SPECIAL_TOKENS.PATH_START)
    path_idx = path_idx_start

    fig, ax = plt.subplots(figsize=figsize)
    mazeplot = None
    camera = Camera(fig)

    while path_idx < len(maze_tokens) + end_offset:
        path_idx += 1

        token_attn: Float[np.ndarray, "subseq_len"] = head_cache[
            maze_id,
            -(len(maze_tokens) - path_idx + 1),
            -len(maze_tokens) :,
        ]

        token_attn = (
            torch.softmax(
                torch.from_numpy(token_attn),
                dim=-1,
            )
            .cpu()
            .numpy()
        )

        target_token: str = maze_tokens[path_idx]

        mazeplot, _, _ = mazeplot_attention(
            maze=maze,
            tokens_context=maze_tokens[:path_idx],
            target=target_token,
            attention=token_attn,
            # colormap_center=0.0,
            cmap="Blues",
            plain_figure=True,
            show_other_tokens=False,
            colormap_max=0.5,
            fig_ax=(fig, ax),
            # hide_colorbar=path_idx > path_idx_start + 1,
            hide_colorbar=False,
            mazeplot=mazeplot,
        )
        camera.snap()

    animation = camera.animate()
    fname_base: str = f"figures/attn_m{maze_id}_H{head_index}L{head_layer}"
    animation.save(f"{fname_base}.gif", fps=fps)
    animation.save(f"{fname_base}.mp4", fps=fps)
