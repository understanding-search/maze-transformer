import warnings

import torch
from jaxtyping import Int
from transformer_lens import HookedTransformer
from zanj.torchutil import ConfigMismatchException, assert_model_cfg_equality

from maze_transformer.training.config import ZanjHookedTransformer


def _check_except_config_equality_modulo_weight_processing(
    diff: dict, model_cfg_keys_allowed_diff: list[str]
) -> bool:
    """given the diff between two configs, return True if the only difference is the specified keys under model_cfg"""
    return all(
        [
            set(diff.keys()) == {"model_cfg"},
            set(diff["model_cfg"].keys()) == {"weight_processing"},
            set(diff["model_cfg"]["weight_processing"]["self"].keys())
            == diff["model_cfg"]["weight_processing"]["other"].keys(),
            all(
                k in model_cfg_keys_allowed_diff
                for k in diff["model_cfg"]["weight_processing"]["self"].keys()
            ),
        ]
    )


class ModelOutputEqualityError(AssertionError):
    """raised when model outputs are not equal"""

    pass


class ModelOutputArgsortEqualityError(ModelOutputEqualityError):
    """raised when argsort of model outputs is not equal"""

    pass


def assert_model_output_equality(
    model_a: HookedTransformer,
    model_b: HookedTransformer,
    vocab_size: int | None = None,
    seq_len_max: int | None = None,
    check_config_equality: bool = True,
    check_argsort_equality: bool = True,
    test_sequence_length: int = 10,
    output_rtol_warn: float = 1e-8,
    output_rtol_assert: float = 1e-4,
):
    """checks the models output the same thing, but first optionally checks that configs are equal (modulo weight processing)"""
    models_are_zanj: bool = isinstance(model_a, ZanjHookedTransformer) and isinstance(
        model_b, ZanjHookedTransformer
    )
    if check_config_equality:
        if not models_are_zanj:
            raise ValueError("cant compare configs of non-zanj models")
        try:
            assert_model_cfg_equality(model_a, model_b)
        except ConfigMismatchException as e:
            if _check_except_config_equality_modulo_weight_processing(
                e.diff, ["are_weights_processed", "are_layernorms_folded"]
            ):
                pass
            else:
                raise e

    # Random input tokens
    if vocab_size is None or seq_len_max is None:
        assert models_are_zanj
        tokenizer = model_a.zanj_model_config.tokenizer
        if vocab_size is None:
            vocab_size = len(tokenizer._token_arr)
        if seq_len_max is None:
            seq_len_max = tokenizer._seq_len_max

    input_sequence: Int[torch.Tensor, "1 test_sequence_length"] = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, min(seq_len_max, test_sequence_length)),
    )

    # (copied from `test_eval_model.py`)
    # Check for equality in argsort (absolute values won't be equal due to centering the unembedding weight matrix)
    if check_argsort_equality:
        output_a_raw: torch.Tensor = model_a(input_sequence.clone())
        output_b_raw: torch.Tensor = model_b(input_sequence.clone())
        output_a_raw_argsort: torch.Tensor = output_a_raw.argsort()
        output_b_raw_argsort: torch.Tensor = output_b_raw.argsort()
        output_argsort_match: torch.Tensor = (
            output_a_raw_argsort == output_b_raw_argsort
        )
        if not torch.all(output_argsort_match):
            raise ModelOutputArgsortEqualityError(
                f"argsort not equal, {output_argsort_match.numel() - output_argsort_match.sum()} / {output_argsort_match.numel()} elements differ",
                f"{vocab_size = }, {test_sequence_length = }",
            )

    # apply normalization (e.g. softmax) and check with atol v-small
    # (roughly 1E-7 for float error on logexp I think)
    output_a: torch.Tensor = torch.nn.functional.softmax(
        model_a(input_sequence.clone()), dim=-1
    )
    output_b: torch.Tensor = torch.nn.functional.softmax(
        model_b(input_sequence.clone()), dim=-1
    )

    if not torch.allclose(output_a, output_b, rtol=output_rtol_warn):
        warnings.warn(
            f"model outputs not equal within rtol={output_rtol_warn}:\n{torch.norm(output_a - output_b) = }"
        )

    if not torch.allclose(output_a, output_b, rtol=output_rtol_assert):
        raise ModelOutputEqualityError(
            f"model outputs not equal within rtol={output_rtol_assert}:\n{torch.norm(output_a - output_b) = }"
        )
