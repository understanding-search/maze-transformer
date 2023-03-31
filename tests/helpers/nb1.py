#%%
# Generic
import typing
import os
from pathlib import Path
import typing
import html

# Plotting
import matplotlib.pyplot as plt

# Numerical Computing
import numpy as np
import torch
import pandas as pd

# Utilities
from muutils.statcounter import StatCounter

# Our Code
from maze_transformer.utils.notebook_utils import configure_notebook
from maze_transformer.generation.latticemaze import LatticeMaze
from maze_transformer.evaluation.plot_maze import PathFormat, MazePlot
from maze_transformer.evaluation.eval_model import MazePath, ArrMazePath, load_model_with_configs, predict_maze_path
from maze_transformer.evaluation.pathdist import MazeEvalFunction, ArrMazeEvalFunction, MazeEvalFuncs, ArrMazeEvalFuncs

# --------------------------------------------------------------------------------

# Disable plots, done during processing by `convert_ipynb_to_script.py`

import matplotlib.pyplot as plt

plt.show = lambda: None

# --------------------------------------------------------------------------------

#%%
# Setup
device = configure_notebook(seed=42, dark_mode=True)

# We won't be training any models
torch.set_grad_enabled(False)

# Get latest model
# this should point towards a directory containing a run. 
# If you don't have any runs, you can create a dataset with `poetry run python scripts/create_dataset.py create ./data/maze 10 --grid_n=4`
# Then train a model with `poetry run python scripts/train_model.py ./data/maze/g4-n10`
run_path = Path("../data/maze/g4-n10/g4-n10_tiny-v1_2023-03-27-17-42-16")

assert run_path.exists(), f"Run path {run_path.as_posix()} does not exist"
model_path = list(sorted(run_path.glob("**/model.final.pt"), key=os.path.getmtime))[
	-1
].resolve()
maze_path = run_path.parent / "maze_tokens.jsonl"

#%%
"""
# plot example mazes
"""

#%%
# setup consts
def testdata_plot_predicted_path(
		model_path: Path,
		maze_tokens_path: Path, 
		n_mazes: int = 10,
		n_tokens_pred: int = 8,
	):
	# load model and configs
	model, cfg = load_model_with_configs(model_path)
	
	# load maze test data - only load first n_mazes
	mazes_tokens: list[list[str]] = []
	with maze_tokens_path.open("r") as f:
		for line in f:
			mazes_tokens.append(line.split(' '))
			if len(mazes_tokens) >= n_mazes:
				break

	mazes_solved: list[tuple[LatticeMaze, MazePath, MazePath]] = [
		predict_maze_path(
			tokens = tokens,
			data_cfg = cfg.dataset_cfg,
			model = model,
		n_tokens_pred = n_tokens_pred,
		)
		for tokens in mazes_tokens
	]


	# plot
	for maze, path, path_pred in mazes_solved:
		MazePlot(maze).add_true_path(path).add_predicted_path(path_pred).show()

testdata_plot_predicted_path(model_path, maze_path, n_mazes=10, n_tokens_pred=16)

#%%
"""
# run path dist eval
"""

#%%
EvalFuncTuple = tuple[typing.Literal["arr", "list"], MazeEvalFunction|ArrMazeEvalFunction]

ALL_PATHDIST_FUNCS: dict[str, EvalFuncTuple] = {
	**{
		name: ("arr", func)
		for name, func in ArrMazeEvalFuncs.__dict__.items()
		if not name.startswith("_")
	},
	**{
		name: ("list", func)
		for name, func in MazeEvalFuncs.__dict__.items()
		if not name.startswith("_")
	},
}

print(ALL_PATHDIST_FUNCS)

#%%
def evaluate_model_pathdist_scores(
		model_path: Path,
		maze_tokens_path: Path,
		pathdist_functions: dict[str, EvalFuncTuple]|None = ALL_PATHDIST_FUNCS,
		n_tokens_pred: int = 8,
		n_mazes: int = 64,
		verbose: bool = False,
	) -> dict[str, StatCounter]:

	mazes_tokens: list[list[str]] = list()

	# load model and configs
	model, cfg = load_model_with_configs(model_path)

	# load maze test data - only load first n_mazes
	mazes_tokens: list[list[str]] = []
	with maze_tokens_path.open("r") as f:
		for line in f:
			mazes_tokens.append(line.split(' '))
			if len(mazes_tokens) >= n_mazes:
				break

	# Predict paths
	mazes_solved: list[tuple[LatticeMaze, MazePath, MazePath]] = list()
	for tokens in mazes_tokens:
		#TODO Batch the maze generation - is padding done by default for the tokenizer in HkTf?
		maze, p_true, p_pred = predict_maze_path(
			tokens = tokens,
			data_cfg = cfg.dataset_cfg,
			model = model,
			n_tokens_pred = n_tokens_pred,
			verbose = verbose,
		)

		mazes_solved.append((maze, p_true, p_pred))

		if verbose:
			print(f"{p_true = }")
			print(f"{p_pred = }")

	# convert paths
	mazes_solved_arrpath: list[tuple[LatticeMaze, ArrMazePath, ArrMazePath]] = [
		(maze, np.array(p_true), np.array(p_pred))
		for maze, p_true, p_pred in mazes_solved
	]

	# evaluate
	pathdist_scores: dict[str, StatCounter] = dict()
	for name, (pathdist_type, pathdist_func) in pathdist_functions.items():
		if pathdist_type == "list":			pathdist_scores[name] = StatCounter(
				pathdist_func(maze, p_true, p_pred)
				for maze, p_true, p_pred in mazes_solved
			)
		elif pathdist_type == "arr":
			pathdist_scores[name] = StatCounter(
				pathdist_func(maze, p_true, p_pred)
				for maze, p_true, p_pred in mazes_solved_arrpath
			)
		else:
			raise ValueError(f"Invalid pathdist_type: {pathdist_type}")

	return pathdist_scores

evaluate_model_pathdist_scores(model_path, maze_path)

#%%
def evaluate_pathdist_scores_checkpoints(
		run_path: Path, # Path to run, not model.final.pt or checkpoints
		maze_tokens_path: Path,
		checkpoint_indices: list[int]|None = None,
		pathdist_functions: dict[str, EvalFuncTuple]|None = ALL_PATHDIST_FUNCS,
		skip_every_nth: int = 1,
		n_tokens_pred: int = 8,
		n_mazes: int = 10,
		verbose: bool = False,
	) -> dict[str, dict[int, StatCounter]]:

	model_checkpoints: list[tuple[int,Path]]
	assert run_path.is_dir(), f"Model path {run_path} is not a directory (expect run directory, not model files)"

	if checkpoint_indices is not None:
		model_checkpoints = list()
		for i in checkpoint_indices:
			mdl_path: Path = Path(run_path) / f"checkpoints/model.iter_{i}.pt"
			if not mdl_path.exists():
				raise ValueError(f"Checkpoint file {mdl_path} does not exist")
			model_checkpoints.append((i, mdl_path))
	else:
		model_checkpoints = [
			(int(mdl_path.stem.split("_")[-1].split(".")[0]), mdl_path)
			for mdl_path in sorted(Path(run_path).glob("checkpoints/model.iter_*.pt"))
		]
	
	print(f"Found {len(model_checkpoints)} checkpoints:\n\t{model_checkpoints = }")

	pathdist_scores_indices: dict[int, dict[str, StatCounter]] = dict()

	for i, mdl_path in model_checkpoints[::skip_every_nth]:

		print(f"# Evaluating checkpoint {i} at {mdl_path}")
		pathdist_scores_indices[i] = evaluate_model_pathdist_scores(
			model_path = mdl_path,
			pathdist_functions = pathdist_functions,
			n_tokens_pred = n_tokens_pred,
			maze_tokens_path = maze_tokens_path,
			n_mazes = n_mazes,
			verbose = verbose,
		)

	return {
		name: {
			index: scores[name]
			for index, scores in pathdist_scores_indices.items()
		}
		for name in pathdist_scores_idx.values().__iter__().__next__() # get first dict's values
	}

data = evaluate_pathdist_scores_checkpoints(
	run_path = run_path,
	maze_tokens_path = maze_path,
	n_mazes = 10,
	# skip_every_nth=10,
	checkpoint_idxs=[64,640064],
	# verbose = True,
)

#%%
def plot_pathdist_scores(
		data: dict[str, dict[int, StatCounter]],
		colors: dict[str, str]|None = None,
		percentile_bounds: tuple[float, float] = (0.4, 0.6),
	):

	if colors is None:
		colors = {
			func_name: f"C{i}"
			for i, func_name in enumerate(data.keys())
		}

	fig, ax = plt.subplots(len(data), 1, figsize = (8, 4 * len(data)))
	fig.subplots_adjust(hspace = 0.5)
		
	for i, (name, scores_indexed) in enumerate(data.items()):
		x = list(scores_indexed.keys())
		y = [
			scores_indexed[i].median()
			for i in x
		]
		ax[i].plot(x, y, label=name, color=colors[name])
		# plot shaded error bars
		y_ub = [
			scores_indexed[i].percentile(percentile_bounds[1])
			for i in x
		]
		y_lb = [
			scores_indexed[i].percentile(percentile_bounds[0])
			for i in x
		]
		ax[i].fill_between(
			x, y_lb, y_ub,
	    	alpha=0.5, 
			edgecolor=colors[name], facecolor=colors[name],
		)

		ax[i].set_title(f"{name}, {percentile_bounds = }")
		ax[i].set_xlabel("Checkpoint")
		ax[i].set_ylabel("score")

	plt.show()

#%%
plot_pathdist_scores(data)
