# Numerical Computing
import pandas as pd
import torch
from fancy_einsum import einsum
from jaxtyping import Float, Int

# TransformerLens imports
from transformer_lens import ActivationCache, HookedTransformer

# model stuff
from maze_transformer.training.config import ZanjHookedTransformer

LArr = Float[torch.Tensor, "samples"]


def residual_stack_to_logit_diff(
    residual_stack: Float[torch.Tensor, "components batch d_model"],
    cache: ActivationCache,
    logit_diff_directions: Float[torch.Tensor, "samples d_model"],
) -> float:
    scaled_residual_stack = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )

    return (
        einsum(
            "... batch d_model, batch d_model -> ...",
            scaled_residual_stack,
            logit_diff_directions,
        )
        / logit_diff_directions.shape[0]
    )


def logit_diff_direct(
    model_logits: Float[torch.Tensor, "samples d_vocab"],
    tokens_correct: Int[torch.Tensor, "samples"],
    tokens_compare_to: Int[torch.Tensor, "samples"] | None = None,
    diff_per_prompt: bool = True,
) -> Float[torch.Tensor, "samples"] | float:
    """based on Neel's explanatory notebook

    https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/main/Exploratory_Analysis_Demo.ipynb

    if `tokens_compare_to` is None, then compare to sum of logits on all other tokens
    """

    # logit on the correct answer token for each sample
    model_logits_on_correct: LArr = torch.gather(
        model_logits, 1, tokens_correct.unsqueeze(1)
    ).squeeze(1)

    output_diff: LArr
    if tokens_compare_to is None:
        # subtract total logits across all other tokens
        all_logits: LArr = torch.sum(model_logits, dim=1)
        output_diff = model_logits_on_correct - (all_logits - model_logits_on_correct)
    else:
        # subtract just the logit on the compare_to token
        logits_compare_to: LArr = torch.gather(
            model_logits, 1, tokens_compare_to.unsqueeze(1)
        ).squeeze(1)
        output_diff = model_logits_on_correct - logits_compare_to

    assert output_diff.shape == tokens_correct.shape

    if diff_per_prompt:
        return output_diff
    else:
        return output_diff.mean().item()

    # return answer_logits / (all_logits - answer_logits)


def logit_diff_residual_stream(
    model: ZanjHookedTransformer,
    cache: ActivationCache,
    tokens_correct: Int[torch.Tensor, "samples"],
    tokens_compare_to: Int[torch.Tensor, "samples"] | None = None,
    directions: bool = False,
) -> float | tuple[float, torch.Tensor]:
    d_vocab: int = model.config.maze_tokenizer.vocab_size
    d_model: int = model.config.model_cfg.d_model

    # embed the whole vocab first
    vocab_tensor: Float[torch.Tensor, "d_vocab"] = torch.arange(
        d_vocab, dtype=torch.long
    )
    vocab_residual_directions: Float[torch.Tensor, "d_vocab d_model"] = (
        model.tokens_to_residual_directions(vocab_tensor)
    )
    # get embedding of answer tokens
    answer_residual_directions = vocab_residual_directions[tokens_correct]
    # get the directional difference between logits and corrent and logits on {all other tokens, comparison tokens}
    logit_diff_directions: Float[torch.Tensor, "samples d_model"]
    if tokens_compare_to is None:
        logit_diff_directions = (
            answer_residual_directions - vocab_residual_directions[~tokens_correct]
        )
    else:
        logit_diff_directions = (
            answer_residual_directions - vocab_residual_directions[tokens_compare_to]
        )

    # get the values from the cache at the last layer and last token
    final_token_residual_stream: Float[torch.Tensor, "samples d_model"] = cache[
        "resid_post", -1
    ][:, -1, :]

    # scaling the values in residual stream with layer norm
    scaled_final_token_residual_stream: Float[torch.Tensor, "samples d_model"] = (
        cache.apply_ln_to_stack(
            final_token_residual_stream,
            layer=-1,
            pos_slice=-1,
        )
    )

    # measure similarity between the logit diff directions and the residual stream at final layer directions
    average_logit_diff: float = (
        torch.dot(
            scaled_final_token_residual_stream.flatten(),
            logit_diff_directions.flatten(),
        )
        / logit_diff_directions.shape[0]
    ).item()

    if directions:
        return average_logit_diff, logit_diff_directions
    else:
        return average_logit_diff


def logits_diff_multi(
    model: HookedTransformer,
    cache: ActivationCache,
    dataset_target_ids: Int[torch.Tensor, "samples"],
    last_tok_logits: Float[torch.Tensor, "samples d_vocab"],
    noise_sigmas: list[float] = [1, 2, 3, 5, 10],
    n_randoms: int = 1,
) -> pd.DataFrame:
    d_vocab: int = last_tok_logits.shape[1]

    test_logits: dict[str, Float[torch.Tensor, "samples"]] = {
        "target": dataset_target_ids,
        "predicted": last_tok_logits.argmax(dim=-1),
        "sampled": torch.multinomial(
            torch.softmax(last_tok_logits, dim=-1), num_samples=1
        ).squeeze(-1),
        **{
            f"noise={s:.2f}": (
                last_tok_logits + s * torch.randn_like(last_tok_logits)
            ).argmax(dim=-1)
            for s in noise_sigmas
        },
        # "random": torch.randint_like(dataset_target_ids, low=0, high=d_vocab),
        **{
            f"random_r{i}": torch.randint_like(dataset_target_ids, low=0, high=d_vocab)
            for i in range(n_randoms)
        },
    }
    compare_dict: dict[str, None | Float[torch.Tensor, "samples"]] = {
        "all": None,
        "random": torch.randint_like(dataset_target_ids, low=0, high=d_vocab),
        "target": dataset_target_ids,
    }

    outputs: list[dict] = list()

    for k_comp, compare_to in compare_dict.items():
        for k, d in test_logits.items():
            result_orig: float = logit_diff_direct(
                model_logits=last_tok_logits,
                tokens_correct=d,
                diff_per_prompt=False,
                tokens_compare_to=compare_to,
            )
            result_res: float = logit_diff_residual_stream(
                model=model,
                cache=cache,
                tokens_correct=d,
                tokens_compare_to=compare_to,
            )
            # print(f"logit diff of {k}\tcompare:\t{'all' if compare_to is None else 'random'}\t{result = }\t{result_res = }")
            outputs.append(
                dict(
                    test=k,
                    compare_to=k_comp,
                    result_orig=result_orig,
                    result_res=result_res,
                )
            )

    df_out: pd.DataFrame = pd.DataFrame(outputs)
    df_out["diff"] = df_out["result_orig"] - df_out["result_res"]
    df_out["ratio"] = df_out["result_orig"] / df_out["result_res"]

    return df_out
