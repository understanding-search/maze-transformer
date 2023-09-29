from typing import NamedTuple, Annotated

# numerical
import numpy as np
from jaxtyping import Float
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# scipy
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
# from scipy.spatial.distance import cosine

# transformerlens
from transformer_lens import HookedTransformer, ActivationCache

# maze_dataset
from maze_dataset.constants import _SPECIAL_TOKENS_ABBREVIATIONS
from maze_dataset.tokenization.token_utils import strings_to_coords
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode

def coordinate_to_color(coord: tuple[float, float], max_val: float = 1.0) -> tuple[float, float, float]:
    """Maps a coordinate (i, j) to a unique RGB color"""
    coord = np.array(coord)
    if max_val < coord.max():
        raise ValueError(f"max_val ({max_val}) must be at least as large as the largest coordinate ({coord.max()})")
    
    coord = coord / max_val

    return (
        coord[0] * 0.6 + 0.3, # r
        0.5,                  # g
        coord[1] * 0.6 + 0.3, # b
    )


TokenPlottingInfo = NamedTuple(
    "TokenPlottingInfo",
    token = str,
    coord = tuple[float, float]|str,
    color = tuple[float, float, float],
)

def process_tokens_for_pca(tokenizer: MazeTokenizer) -> list[TokenPlottingInfo]:

    tokens_coords: list[str|tuple[int,int]] = strings_to_coords(tokenizer.token_arr, when_noncoord="include")
    tokens_coords_only: list[tuple[int,int]] = strings_to_coords(tokenizer.token_arr, when_noncoord="skip")
    max_coord: int = np.array(tokens_coords_only).max()
    # token_idxs_coords: list[int] = tokenizer.encode(tokenizer.coords_to_strings(tokens_coords_only))

    vocab_coordinates_colored: list[TokenPlottingInfo] = [
        TokenPlottingInfo(*x) for x in        
        zip(
            tokenizer.token_arr,
            tokens_coords,
            [
                coordinate_to_color(coord, max_val=max_coord) if isinstance(coord, tuple) else (0.0, 1.0, 0.0)
                for coord in tokens_coords
            ],
        )
    ]
        
    return vocab_coordinates_colored

EmbeddingsPCAResult = NamedTuple(
    "EmbeddingsPCAResult",
    result = np.ndarray,
    index_map = list[int]|None,
    pca_obj = PCA,
)

def compute_pca(
        model: HookedTransformer, 
        token_plotting_info: list[TokenPlottingInfo],
    ) -> dict[str, EmbeddingsPCAResult]:

    pca_all: PCA = PCA(svd_solver='full')
    pca_coords: PCA = PCA(svd_solver='full')
    pca_special: PCA = PCA(svd_solver='full')

    # PCA_RESULTS = pca_all.fit_transform(MODEL.W_E.cpu().numpy().T)
    # PCA_RESULTS_COORDS_ONLY = pca_coords.fit_transform(MODEL.W_E[token_idxs_coords].cpu().numpy().T)

    idxs_coords: list[int] = list()
    idxs_special: list[int] = list()

    i: int; tokinfo: TokenPlottingInfo
    for i, tokinfo in enumerate(token_plotting_info):
        if isinstance(tokinfo.coord, tuple):
            idxs_coords.append(i)
        elif isinstance(tokinfo.coord, str):
            idxs_special.append(i)
        else:
            raise ValueError(f"unexpected coord type: {type(tokinfo.coord)}\n{tokinfo = }")

    return dict(
        all = EmbeddingsPCAResult(
            result = pca_all.fit_transform(model.W_E.cpu().numpy().T), 
            index_map = None,
            pca_obj = pca_all,
        ),
        coords_only = EmbeddingsPCAResult(
            result = pca_coords.fit_transform(model.W_E[idxs_coords].cpu().numpy().T), 
            index_map = idxs_coords,
            pca_obj = pca_coords,
        ),
        special_only = EmbeddingsPCAResult(
            result = pca_special.fit_transform(model.W_E[idxs_special].cpu().numpy().T), 
            index_map = idxs_special,
            pca_obj = pca_special,
        ),
    )


def plot_pca_colored(
    pca_results_options: dict[str, EmbeddingsPCAResult],
    pca_results_key: str,
    vocab_colors: list[tuple],
    dim1: int, 
    dim2: int,
    lattice_connections: bool = True,
    symlog_scale: float|None = None,
    axes_and_centered: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    
    # set up figure, get PCA results
    fig, ax = plt.subplots(figsize=(5, 5))
    pca_result: EmbeddingsPCAResult = pca_results_options[pca_results_key]

    # Store lattice points for drawing connections
    lattice_points: tuple[tuple[int, int], tuple[float, float]] = list()

    for i in range(pca_result.result.shape[1]):
        # map index if necessary
        if pca_result.index_map is not None:
            i_map: int = pca_result.index_map[i]
        else:
            i_map = i
        token, coord, color = vocab_colors[i_map]
        # plot the point
        ax.scatter(
            pca_result.result[dim1-1, i],
            pca_result.result[dim2-1, i],
            alpha=0.5,
            color=color,
        )
        if isinstance(coord, str):
            # label with the abbreviated token name
            ax.text(
                pca_result.result[dim1-1, i], 
                pca_result.result[dim2-1, i], 
                _SPECIAL_TOKENS_ABBREVIATIONS[coord],
                fontsize=8,
            )
        else:
            # add to the lattice points list for later
            lattice_points.append((
                coord,
                (pca_result.result[dim1-1, i], pca_result.result[dim2-1, i]),
            ))
            
    if axes_and_centered:
        # find x and y limits
        xbound: float = np.max(np.abs(pca_result.result[dim1-1])) * 1.1
        ybound: float = np.max(np.abs(pca_result.result[dim2-1])) * 1.1
        # set axes limits
        ax.set_xlim(-xbound, xbound)
        ax.set_ylim(-ybound, ybound)
        # plot axes
        ax.plot([-xbound, xbound], [0, 0], color='black', alpha=0.5, linewidth=0.5)
        ax.plot([0, 0], [-ybound, ybound], color='black', alpha=0.5, linewidth=0.5)
    
    # add lattice connections
    if lattice_connections:
        for (i, j), (x, y) in lattice_points:
            for (i2, j2), (x2, y2) in lattice_points:
                # manhattan distance of 1
                if np.linalg.norm(np.array([i, j]) - np.array([i2, j2]), ord=1) == 1:
                    # plot a line between the two points
                    ax.plot(
                        [x, x2],
                        [y, y2],
                        color='red',
                        alpha=0.2,
                        linewidth=0.5,
                    )
        
    ax.set_xlabel(f"PC{dim1}")
    ax.set_ylabel(f"PC{dim2}")
    ax.set_title(f"PCA of Survey Responses:\nPC{dim1} vs PC{dim2}")

    # semi-log scale
    if isinstance(symlog_scale, (float, int)):
        if symlog_scale > 0:
            ax.set_xscale('symlog', linthresh=symlog_scale)
            ax.set_yscale('symlog', linthresh=symlog_scale)

    return fig, ax


def compute_distances_and_correlation(
        embedding_matrix: Float[np.ndarray, "d_vocab d_model"], 
        tokenizer: MazeTokenizer,
        embedding_metric: str = "cosine",
        coordinate_metric: str = "euclidean",
        show: bool = True,
    ) -> dict:

    coord_tokens_ids: dict[str, int] = tokenizer.coordinate_tokens_ids
    coord_embeddings: Float[np.ndarray, "n_coord_tokens d_model"] = np.array([
        embedding_matrix[v]
        for v in coord_tokens_ids.values()
    ])
    
    # Calculate the pairwise distances in embedding space
    embedding_distances: Float[np.ndarray, "n_coord_tokens d_model"] = pdist(
        coord_embeddings, 
        metric=embedding_metric,
    )
    # normalize the distance by the maximum distance
    embedding_distances /= embedding_distances.max()

    # Convert the distances to a square matrix
    embedding_distances_matrix: Float[np.ndarray, "n_coord_tokens n_coord_tokens"] = squareform(embedding_distances)

    # Calculate the correlation between the embedding and coordinate distances
    coordinate_coordinates: Float[np.ndarray, "n_coord_tokens 2"] = np.array(list(tokenizer.coordinate_tokens_coords.keys()))
    coordinate_distances = pdist(
        coordinate_coordinates, 
        metric=coordinate_metric,
    )
    correlation, corr_pval = pearsonr(embedding_distances, coordinate_distances)

    return dict(
        embedding_distances_matrix = embedding_distances_matrix,
        correlation = correlation,
        corr_pval = corr_pval,
        tokenizer = tokenizer,
        embedding_metric = embedding_metric,
        coordinate_metric = coordinate_metric,
    )
    
def plot_distances_matrix(
        embedding_distances_matrix: Float[np.ndarray, "n_coord_tokens n_coord_tokens"],
        tokenizer: MazeTokenizer,
        embedding_metric: str,
        show: bool = True,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:

    coord_tokens_ids: dict[str, int] = tokenizer.coordinate_tokens_ids

    # Plot the embedding distances
    fig, ax = plt.subplots(figsize=(15, 15))
    cax = ax.matshow(
        embedding_distances_matrix, 
        cmap='viridis',
        interpolation='none',
    )
    ax.grid(which='major', color='white', linestyle='-', linewidth=0.5)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.0)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(coord_tokens_ids)))
    ax.set_yticks(np.arange(len(coord_tokens_ids)))
    ax.set_xticklabels(coord_tokens_ids.keys())
    ax.set_yticklabels(coord_tokens_ids.keys())

    plt.setp(ax.get_xticklabels(), rotation=90, ha="left", rotation_mode="anchor")

    ax.set_title(f"{embedding_metric} Distances Between Coordinate Embeddings")
    ax.grid(False)

    if show:
        plt.show()

    return fig, ax

def plot_distance_grid(
        embedding_distances_matrix: Float[np.ndarray, "n_coord_tokens n_coord_tokens"], 
        tokenizer: MazeTokenizer,
        embedding_metric: str,
        coordinate_metric: str,
        show: bool = True,
        **kwargs,
    ):
    n: int = tokenizer.max_grid_size
    # print(n)
    # print(tokenizer.coordinate_tokens_coords)
    fig, axs = plt.subplots(n, n, figsize=(20, 20))

    for idx, ((x, y), token_id) in enumerate(tokenizer.coordinate_tokens_coords.items()):
        ax = axs[x, y]
        
        # Extract distances for this particular token from the distance matrix
        distances: Float[np.ndarray, "n_coord_tokens"] = embedding_distances_matrix[idx, :]
        
        # get distances
        grid_distances: Float[np.ndarray, "n n"] = np.full((n, n), np.nan)
        for (x2, y2), distance in zip(tokenizer.coordinate_tokens_coords.keys(), distances):
            grid_distances[x2, y2] = distance
        # coords = np.array(list(tokenizer.coordinate_tokens_coords.keys()))
        # grid_distances[coords[:, 0], coords[:, 1]] = distances

        cax = ax.matshow(grid_distances, cmap='viridis', interpolation='none')
        ax.plot(y, x, 'rx')
        ax.set_title(f"from ({x},{y})")
        # fully remove both major and minor gridlines
        ax.grid(False)

    fig.suptitle(f"{embedding_metric} distances grid")
    plt.colorbar(cax, ax=axs.ravel().tolist())

    if show:
        plt.show()

def plot_correlation(
    embedding_distances_matrix: Float[np.ndarray, "n_coord_tokens n_coord_tokens"],
    tokenizer: MazeTokenizer,
    correlation: float,
    corr_pval: float,
    embedding_metric: str,
    coordinate_metric: str,
    show: bool = True,
):
    raise NotImplementedError()