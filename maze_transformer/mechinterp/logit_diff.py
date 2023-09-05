# Generic
import os
from pathlib import Path
from copy import deepcopy
import typing

# Numerical Computing
import numpy as np
import torch
import pandas as pd
# import torch.nn.functional as F
from fancy_einsum import einsum
import einops
from jaxtyping import Float, Int, Bool
import matplotlib.pyplot as plt

from muutils.misc import shorten_numerical_to_str
from muutils.nbutils.configure_notebook import configure_notebook
# TransformerLens imports
from transformer_lens import ActivationCache

# Our Code
# dataset stuff
from maze_dataset import MazeDataset, MazeDatasetConfig, SolvedMaze, LatticeMaze, SPECIAL_TOKENS
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode
from maze_dataset.plotting.print_tokens import color_maze_tokens_AOTP

# model stuff
from maze_transformer.training.config import ConfigHolder, ZanjHookedTransformer, BaseGPTConfig

def residual_stack_to_logit_diff(
        residual_stack: Float[torch.Tensor, "components batch d_model"], 
        cache: ActivationCache,
        logit_diff_directions: Float[torch.Tensor, "n_mazes d_model"],
    ) -> float:

    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer = -1, pos_slice=-1)

    return einsum(
        "... batch d_model, batch d_model -> ...", 
        scaled_residual_stack, logit_diff_directions,
    ) / logit_diff_directions.shape[0]


def logit_diff_orig(
        final_logits: Float[torch.Tensor, "n_mazes d_vocab"],
        answer_tokens: Int[torch.Tensor, "n_mazes"],
        compare_to: Int[torch.Tensor, "n_mazes"]|None = None,
        per_prompt: bool = True,
    ) -> Float[torch.Tensor, "n_mazes"]|float:
    """From Neels explanatory notebook
    
    https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/main/Exploratory_Analysis_Demo.ipynb
    """
    LArr = Float[torch.Tensor, "n_mazes"]

    # logit on the answer token for each sample
    answer_logits: LArr = torch.gather(final_logits, 1, answer_tokens.unsqueeze(1)).squeeze(1)
 
    output: LArr
    if compare_to is None:
        # logits of all tokens for each sample
        all_logits: LArr = torch.sum(final_logits, dim=1)
        output = answer_logits - (all_logits - answer_logits)
    else:
        # specifically the comparison tokens
        compare_to_logits: LArr = torch.gather(final_logits, 1, compare_to.unsqueeze(1)).squeeze(1)
        output = answer_logits - compare_to_logits

    assert output.shape == answer_tokens.shape

    if per_prompt:
        return output
    else:
        return output.mean().item()

    # return answer_logits / (all_logits - answer_logits)


def logit_diff_residual_stream(
	model: ZanjHookedTransformer,
	cache: ActivationCache,
	answer_tokens: Int[torch.Tensor, "n_mazes"],
	compare_to: Int[torch.Tensor, "n_mazes"]|None = None,
	directions: bool = False,
) -> float|tuple[float, torch.Tensor]:
	# embed the whole vocab first
	d_vocab: int = model.zanj_model_config.maze_tokenizer.vocab_size
	vocab_tensor: Float[torch.Tensor, "d_vocab"] = torch.arange(d_vocab, dtype=torch.long)
	vocab_residual_directions = model.tokens_to_residual_directions(vocab_tensor)
	# get embedding of answer tokens
	answer_residual_directions = vocab_residual_directions[answer_tokens]
	# get the directional difference
	logit_diff_directions: Float[torch.Tensor, "n_mazes d_model"]
	if compare_to is None:
		logit_diff_directions = answer_residual_directions - vocab_residual_directions[~answer_tokens]
	else:
		logit_diff_directions = answer_residual_directions - vocab_residual_directions[compare_to]


	# get the values from the cache at the last layer and last token
	final_token_residual_stream = cache["resid_post", -1][:, -1, :]
	# scaling the values in residual stream with layer norm
	scaled_final_token_residual_stream = cache.apply_ln_to_stack(
		final_token_residual_stream, layer = -1, pos_slice=-1,
	)


	average_logit_diff = torch.dot(
		scaled_final_token_residual_stream.flatten(),
		logit_diff_directions.flatten(),
	) / logit_diff_directions.shape[0]

	if directions:
		return average_logit_diff.item(), logit_diff_directions
	else:
		return average_logit_diff.item()

def logits_diff_multi(
	model: ZanjHookedTransformer,
	cache: ActivationCache,
	dataset_target_ids: Int[torch.Tensor, "n_mazes"],
	last_tok_logits: Float[torch.Tensor, "n_mazes d_vocab"],
	noise_sigmas: list[float] = [1, 2, 3, 5, 10],
	n_randoms: int = 1,
) -> pd.DataFrame:
	d_vocab: int = last_tok_logits.shape[1]
	
	test_logits: dict[str, Float[torch.Tensor, "n_mazes"]] = {
		"target": dataset_target_ids, 
		"predicted": last_tok_logits.argmax(dim=-1), 
		"sampled": torch.multinomial(torch.softmax(last_tok_logits, dim=-1), num_samples=1).squeeze(-1),
		**{
			f"noise={s:.2f}": (last_tok_logits + s*torch.randn_like(last_tok_logits)).argmax(dim=-1)
			for s in noise_sigmas
		},
		# "random": torch.randint_like(dataset_target_ids, low=0, high=d_vocab),
		**{
			f"random_r{i}": torch.randint_like(dataset_target_ids, low=0, high=d_vocab)
			for i in range(n_randoms)
		},
	}
	compare_dict: dict[str, None|Float[torch.Tensor, "n_mazes"]] = {
		"all": None,
		"random": torch.randint_like(dataset_target_ids, low=0, high=d_vocab),
		"target": dataset_target_ids,
	}

	outputs: list[dict] = list()

	for k_comp, compare_to in compare_dict.items():
		for k, d in test_logits.items():
			result_orig: float = logit_diff_orig(
				final_logits=last_tok_logits, 
				answer_tokens=d,
				per_prompt=False,
				compare_to=compare_to,
			)
			result_res: float = logit_diff_residual_stream(
				model=model,
				cache=cache,
				answer_tokens=d,
				compare_to=compare_to,
			)
			# print(f"logit diff of {k}\tcompare:\t{'all' if compare_to is None else 'random'}\t{result = }\t{result_res = }")
			outputs.append(dict(
				test=k,
				compare_to=k_comp,
				result_orig=result_orig,
				result_res=result_res,
			))

	df_out: pd.DataFrame = pd.DataFrame(outputs)
	df_out["diff"] = df_out["result_orig"] - df_out["result_res"]
	df_out["ratio"] = df_out["result_orig"] / df_out["result_res"]


	return df_out
