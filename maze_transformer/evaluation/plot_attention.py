# Generic
import typing

# plotting
import IPython
import matplotlib
import matplotlib.pyplot as plt

# Numerical Computing
import numpy as np
import torch

# Transformers
from circuitsvis.attention import attention_heads
from circuitsvis.tokens import colored_tokens_multi
from jaxtyping import Float
from maze_dataset import CoordTup, MazeDataset, MazeDatasetConfig, SolvedMaze
from maze_dataset.maze.lattice_maze import coord_str_to_tuple_noneable
from maze_dataset.plotting import MazePlot

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
            tokens: list[str] = solved_maze.as_tokens(dataset.cfg.node_token_map)
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


def colorize(
    tokens: list[str],
    weights: list[float],
    cmap: matplotlib.colors.Colormap | str = "Blues",
    template: str = '<span class="barcode"; style="color: black; background-color: {clr}">&nbsp{tok}&nbsp</span>',
) -> str:
    """given a sequence of tokens and their weights, colorize the tokens according to the weights (output is html)

    originally from https://stackoverflow.com/questions/59220488/to-visualize-attention-color-tokens-using-attention-weights
    """

    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)

    colored_string: str = ""

    for word, color in zip(tokens, weights):
        color_hex: str = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(clr=color_hex, tok=word)

    return colored_string


def _test():
    mystr: str = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est laborum"
    tokens: list[str] = mystr.split()
    weights: list[float] = np.random.rand(len(tokens)).tolist()
    colored: str = colorize(tokens, weights)
    IPython.display.display(IPython.display.HTML(colored))
