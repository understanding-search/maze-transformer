from typing import NamedTuple, Annotated

# numerical
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
) -> None:
	
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
                        color='black',
                        alpha=0.5,
                        linewidth=0.5,
                    )
        
    ax.set_xlabel(f"PC{dim1}")
    ax.set_ylabel(f"PC{dim2}")
    ax.set_title(f"PCA of Survey Responses:\nPC{dim1} vs PC{dim2}")
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()