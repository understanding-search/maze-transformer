import warnings

import torch
from jaxtyping import Int
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


def assert_model_output_equality(
    model_a: ZanjHookedTransformer,
    model_b: ZanjHookedTransformer,
    test_sequence_length: int = 10,
    output_rtol_warn: float = 1e-8,
    output_rtol_assert: float = 1e-4,
):
    """checks that configs are equal (modulo weight processing) and that the models output the same thing"""
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
    tokenizer = model_a.zanj_model_config.tokenizer
    input_sequence: Int[torch.Tensor, "1 test_sequence_length"] = torch.randint(
        low=0,
        high=len(tokenizer._token_arr),
        size=(1, min(tokenizer._seq_len_max, test_sequence_length)),
    )

    # (copied from `test_eval_model.py`)
    # Check for equality in argsort (absolute values won't be equal due to centering the unembedding weight matrix)
    assert torch.all(
        model_a(input_sequence.clone()).argsort()
        == model_b(input_sequence.clone()).argsort()
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

    assert torch.allclose(
        output_a, output_b, rtol=output_rtol_assert
    ), f"model outputs not equal within rtol={output_rtol_assert}:\n{torch.norm(output_a - output_b) = }"
