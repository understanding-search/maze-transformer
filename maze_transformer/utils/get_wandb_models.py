import itertools
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import wandb
from maze_dataset import MazeDatasetConfig
from maze_dataset.tokenization import TokenizationMode
from muutils.misc import sanitize_fname, shorten_numerical_to_str
from muutils.tensor_utils import (
    StateDictKeysError,
    StateDictShapeError,
    StateDictValueError,
    compare_state_dicts,
)
from transformer_lens import HookedTransformer
from wandb.sdk.wandb_run import Artifact, Run
from zanj import ZANJ
from zanj.torchutil import ConfigMismatchException, assert_model_cfg_equality

from maze_transformer.test_helpers.assertions import (
    ModelOutputArgsortEqualityError,
    ModelOutputEqualityError,
    _check_except_config_equality_modulo_weight_processing,
    assert_model_output_equality,
)
from maze_transformer.training.config import (
    BaseGPTConfig,
    ConfigHolder,
    TrainConfig,
    ZanjHookedTransformer,
)

# from rich import print


def get_step(
    artifact: Artifact, step_prefix: str = "step=", except_if_invalid: bool = False
) -> int:
    step_alias: list[str] = [
        alias for alias in artifact.aliases if alias.startswith(step_prefix)
    ]
    if len(step_alias) != 1:  # if we have multiple, skip as well
        if except_if_invalid:
            raise KeyError(
                f"Could not find step alias in {artifact.name} " f"{artifact.aliases}",
            )
        else:
            return -1
    return int(step_alias[0].replace(step_prefix, ""))


def load_model(
    config_holder: ConfigHolder, model_path: str, fold_ln: bool = True
) -> HookedTransformer:
    model: HookedTransformer = config_holder.create_model()
    state_dict: dict = torch.load(model_path, map_location=model.cfg.device)
    model.load_and_process_state_dict(
        state_dict,
        fold_ln=False,
        center_writing_weights=True,
        center_unembed=True,
        refactor_factored_attn_matrices=True,
    )
    model.process_weights_(fold_ln=fold_ln)
    model.setup()  # Re-attach layernorm hooks by calling setup
    model.eval()
    return model


def match_checkpoint(
    checkpoint: int | None,
    run: Run,
    step_prefix: str = "step=",
) -> Artifact:
    # Match checkpoint
    # available_checkpoints = [
    #     artifact for artifact in run.logged_artifacts() if artifact.type == "model"
    # ]
    available_checkpoints: list[Artifact] = list(run.logged_artifacts())
    artifact: list[Artifact] = [
        aft for aft in available_checkpoints if get_step(aft, step_prefix) == checkpoint
    ]
    if len(artifact) != 1:
        raise KeyError(
            f"Could not find checkpoint {checkpoint} in {run.name} "
            f"Available checkpoints: ",
            str(
                [
                    f"{artifact.name} | steps: {get_step(artifact, step_prefix)}"
                    for artifact in available_checkpoints
                ]
            ),
            "\n",
            str([(x.name, x.aliases) for x in available_checkpoints]),
        )

    artifact = artifact[0]
    print(f"Loading checkpoint {checkpoint}")
    return artifact


def load_wandb_run(
    project: str = "aisc-search/alex",
    run_id: str = "sa973hyn",
    output_path: str = "./downloaded_models",
    checkpoint: int | None = None,
) -> tuple[HookedTransformer, ConfigHolder]:
    api: wandb.Api = wandb.Api()

    artifact_name: str = f"{project.rstrip('/')}/{run_id}"

    run: Run = api.run(artifact_name)
    wandb_cfg: wandb.config.Config = run.config  # Get run configuration

    # -- Get / Match checkpoint --
    artifact: Artifact
    if checkpoint is not None:
        artifact = match_checkpoint(checkpoint, run)
    else:
        # Get latest checkpoint
        print("Loading latest checkpoint")
        artifact_name = f"{artifact_name}:latest"
        artifact = api.artifact(artifact_name)
        checkpoint = get_step(artifact)

    # -- Initalize configurations --
    # Model cfg
    model_properties = {
        k: wandb_cfg[k] for k in ["act_fn", "d_model", "d_head", "n_layers"]
    }
    model_cfg: BaseGPTConfig = BaseGPTConfig(
        name=f"model {run_id}",
        weight_processing={
            "are_layernorms_folded": False,
            "are_weights_processed": False,
        },
        **model_properties,
    )

    # Dataset cfg
    grid_n: int = math.sqrt(wandb_cfg["d_vocab"] - 11)  #! Jank
    assert grid_n == int(
        grid_n
    ), "grid_n must be a perfect square + 11"  # check integer
    ds_cfg: MazeDatasetConfig = MazeDatasetConfig(
        name=wandb_cfg.get("dataset_name", "no_name"), grid_n=int(grid_n), n_mazes=-1
    )

    cfg: ConfigHolder = ConfigHolder(
        model_cfg=model_cfg,
        dataset_cfg=ds_cfg,
        train_cfg=TrainConfig(
            name=f"artifact '{artifact_name}', checkpoint '{checkpoint}'"
        ),
    )
    download_path: Path = (
        Path(output_path)
        / f'{artifact.name.split(":")[0]}'
        / f"model.iter_{checkpoint}.pt"
    )
    #! Account for final checkpoint
    if not download_path.exists():
        artifact.download(root=download_path.parent)
        print(f"Downloaded model to {download_path}")
    else:
        print(f"Model already downloaded to {download_path}")

    print("Loading model")
    model: HookedTransformer = load_model(cfg, download_path, fold_ln=True)
    return model, cfg


def load_and_configure_wandb_model(
    run_id: str,
    project: str = "aisc-search/alex",
    checkpoint: int | None = None,
    tokenization_mode_override: TokenizationMode | None = None,
) -> tuple[HookedTransformer, ConfigHolder, dict]:
    wandb_kwargs: dict = dict(
        project=project,
        run_id=run_id,
        checkpoint=checkpoint,
    )
    model_wandb: HookedTransformer
    cfg: ConfigHolder
    model_wandb, cfg = load_wandb_run(**wandb_kwargs)

    if tokenization_mode_override is not None:
        print(
            f"# Overriding tokenization mode with {tokenization_mode_override = }, original: {cfg.maze_tokenizer.tokenization_mode = }"
        )
        cfg.maze_tokenizer.tokenization_mode = tokenization_mode_override

    print(f"\t{cfg.model_cfg.weight_processing = }")
    print(f"\t{type(model_wandb) = } {type(cfg) = }")

    return model_wandb, cfg, wandb_kwargs


def convert_model_to_zanj(
    model: HookedTransformer,
    cfg: ConfigHolder,
    wandb_kwargs: dict,
) -> ZanjHookedTransformer:
    model_zanj: ZanjHookedTransformer = ZanjHookedTransformer(cfg)
    model_zanj.load_state_dict(model.state_dict())
    model_zanj.training_records = {
        "load_wandb_run_kwargs": wandb_kwargs,
        "train_cfg.name": cfg.train_cfg.name,
    }
    print(
        f"\tgot zanj model with {shorten_numerical_to_str(model_zanj.num_params())} parameters"
    )
    print(f"\t{model_zanj.training_records = }")
    print(f"\t{model_zanj.zanj_model_config.model_cfg.weight_processing = }")

    return model_zanj


def full_model_compare(
    model_a: HookedTransformer,
    model_b: HookedTransformer,
    cfg: ConfigHolder,
) -> tuple[dict[str, bool], dict[str, Any]]:
    """will pass on config tests if non-zanj model"""
    vocab_size: int = cfg.maze_tokenizer.vocab_size
    seq_len_max: int = cfg.dataset_cfg.seq_len_max
    tests_passed: dict[str, bool | None] = dict(
        zanj_config=False,
        zanj_config_nowp=False,
        ht_config=False,
        output_argsort=False,
        output_no_argsort=False,
        state_dict_keys=False,
        state_dict_shape=False,
        state_dict_value=False,
    )
    tests_info: dict[str, Any] = dict()

    # check hooked transformer config
    if asdict(model_a.cfg) == asdict(model_b.cfg):
        tests_passed["ht_config"] = True

    # check zanj config, if possible
    if isinstance(model_a, ZanjHookedTransformer) and isinstance(
        model_b, ZanjHookedTransformer
    ):
        try:
            assert_model_cfg_equality(model_a, model_b)
            tests_passed["zanj_config"] = True
            tests_passed["zanj_config_nowp"] = True
        except ConfigMismatchException as e:
            tests_info["zanj_config"] = str(e)
            if _check_except_config_equality_modulo_weight_processing(
                e.diff, ["are_weights_processed", "are_layernorms_folded"]
            ):
                tests_passed["zanj_config_nowp"] = True
    else:
        # why a string here instead of `None`? so that we can do `if tests_passed["zanj_config"]`
        tests_passed["zanj_config"] = "not_zanj"
        tests_passed["zanj_config_nowp"] = "not_zanj"

    # check output equality
    try:
        assert_model_output_equality(
            model_a,
            model_b,
            check_config_equality=False,
            check_argsort_equality=True,
            vocab_size=vocab_size,
            seq_len_max=seq_len_max,
        )
        tests_passed["output_argsort"] = True
        tests_passed["output_no_argsort"] = True
    except ModelOutputArgsortEqualityError as e:
        tests_info["output_argsort"] = e
        # if argsort fails, try again with argsort check disabled
        try:
            assert_model_output_equality(
                model_a,
                model_b,
                check_config_equality=False,
                check_argsort_equality=False,
                vocab_size=vocab_size,
                seq_len_max=seq_len_max,
            )
            tests_passed["output_no_argsort"] = True
        except ModelOutputEqualityError as e:
            tests_info["output_no_argsort"] = e

    # check state dict equality
    try:
        compare_state_dicts(model_a.state_dict(), model_b.state_dict())
        tests_passed["state_dict_keys"] = True
        tests_passed["state_dict_shape"] = True
        tests_passed["state_dict_value"] = True
    except StateDictKeysError as e:
        tests_info["state_dict_shape"] = e
    except StateDictShapeError as e:
        tests_passed["state_dict_keys"] = True
        tests_info["state_dict_shape"] = e
    except StateDictValueError as e:
        tests_passed["state_dict_keys"] = True
        tests_passed["state_dict_shape"] = True
        tests_info["state_dict_value"] = e

    return tests_passed, tests_info


ModelComboTestResults = dict[
    tuple[str, str],  # model names
    tuple[
        dict[str, bool],  # tests passed
        dict[str, Any],  # tests info
    ],
]


def compare_model_combos(
    model_dict: dict[str, HookedTransformer],
    cfg: ConfigHolder,
    verbose: bool = True,
) -> ModelComboTestResults:
    output: ModelComboTestResults = dict()

    for (name_a, m_a), (name_b, m_b) in itertools.combinations(model_dict.items(), 2):
        if verbose:
            print(f"\tcomparing: {name_a}, {name_b}")

        tests_passed, tests_info = full_model_compare(m_a, m_b, cfg)
        output[(name_a, name_b)] = tests_passed, tests_info

    return output


def perform_reload_checks(
    model_wandb: HookedTransformer,
    cfg: ConfigHolder,
    model_zanj: ZanjHookedTransformer,
    model_path: str,
    verbose: bool = True,
) -> bool:
    print(f"# Reloading model from {model_path.as_posix()}")
    model_loaded: ZanjHookedTransformer = ZanjHookedTransformer.read(model_path)
    print(f"\t{model_loaded.zanj_model_config.model_cfg.weight_processing = }")
    # fold layernorms for wandb model
    model_loaded_process_weights: ZanjHookedTransformer = ZanjHookedTransformer.read(
        model_path
    )
    model_loaded_process_weights.process_weights_(
        fold_ln=True, center_writing_weights=False, center_unembed=False
    )

    model_dict: dict[str, HookedTransformer] = {
        "wandb": model_wandb,
        "zanj": model_zanj,
        "zanj_loaded": model_loaded,
        "zanj_loaded_processed": model_loaded_process_weights,
    }

    compare_result: ModelComboTestResults = compare_model_combos(
        model_dict=model_dict,
        cfg=cfg,
        verbose=verbose,
    )

    print(f"# Comparison results:")
    outputs_keys: list[str] = ["output_argsort", "output_no_argsort"]
    for (model_a, model_b), (test_results, test_info) in compare_result.items():
        print(f"\t## {model_a} vs {model_b}")
        if not all(test_results.values()):
            failed_tests: list[str] = [k for k, v in test_results.items() if not v]
            print(f"\tFAILED: {failed_tests}")
            print(
                f"\t!FAILED OUTPUTS: {[k for k in failed_tests if k in outputs_keys]}"
            )
        print(f"\t\t{test_results = }")
        if test_info:
            print(f"\t\t{test_info = }")

    return


def load_wandb_pt_model_as_zanj(
    run_id: str,
    project: str = "aisc-search/alex",
    checkpoint: int | None = None,
    output_path: str = "./downloaded_models",
    save_zanj_model: bool = True,
    verbose: bool = True,
    allow_weight_processing_diff: bool = True,
    tokenization_mode_override: TokenizationMode | None = None,
    test_reload: bool = True,
) -> ZanjHookedTransformer:
    print(
        f"# Loading model and config from wandb:\n{run_id = }, {project = }, {checkpoint = }"
    )
    model_wandb: HookedTransformer
    cfg: ConfigHolder
    wandb_kwargs: dict
    model_wandb, cfg, wandb_kwargs = load_and_configure_wandb_model(
        run_id=run_id,
        project=project,
        checkpoint=checkpoint,
        tokenization_mode_override=tokenization_mode_override,
    )

    print(f"# Converting model to zanj")
    model_zanj: ZanjHookedTransformer = convert_model_to_zanj(
        model=model_wandb,
        cfg=cfg,
        wandb_kwargs=wandb_kwargs,
    )

    print(f"# Checking zanj-converted model matches wandb model")
    assert_model_output_equality(
        model_wandb,
        model_zanj,
        check_config_equality=False,
        vocab_size=cfg.maze_tokenizer.vocab_size,
        seq_len_max=cfg.dataset_cfg.seq_len_max,
    )
    print(f"\tmodel outputs match")
    compare_state_dicts(model_wandb.state_dict(), model_zanj.state_dict())
    print(f"\tstate dicts match")

    zanj: ZANJ = ZANJ(
        custom_settings={
            "_load_state_dict_wrapper": {
                "recover_exact": True,
                "fold_ln": False,
                "refactor_factored_attn_matrices": False,
            }
        }
    )

    if save_zanj_model:
        model_zanj_save_path: Path = (
            Path(output_path) / f"wandb.{wandb_kwargs['run_id']}.zanj"
        )
        print(f"# Saving model to {model_zanj_save_path.as_posix()}")
        model_zanj.save(model_zanj_save_path)

    if test_reload:
        assert save_zanj_model, f"must save model to test reloading"
        perform_reload_checks(
            model_wandb=model_wandb,
            cfg=cfg,
            model_zanj=model_zanj,
            model_path=model_zanj_save_path,
            verbose=verbose,
            # allow_weight_processing_diff=allow_weight_processing_diff,
        )

    print(f"# Checks complete!")

    return model_zanj


def load_wandb_zanj(
    run_id: str,
    project: str = "aisc-search/alex",
    checkpoint: int | None = None,
    output_path: str | Path = "./downloaded_models",
) -> ZanjHookedTransformer:
    output_path = Path(output_path)
    api: wandb.Api = wandb.Api()
    artifact_name: str = f"{project.rstrip('/')}/{run_id}"
    run: Run = api.run(artifact_name)
    wandb_cfg: wandb.config.Config = run.config  # Get run configuration

    print(f"Get artifact from {artifact_name = } corresponding to {checkpoint = }")
    artifact: Artifact
    is_final: bool
    if checkpoint is not None:
        is_final = False
        artifact = match_checkpoint(checkpoint, run, step_prefix="iter-")
    else:
        is_final = True
        # Get latest checkpoint
        print("Loading latest checkpoint")
        artifact_name = f"{artifact_name}:latest"
        artifact = api.artifact(artifact_name)
        checkpoint = get_step(artifact, step_prefix="iter-")
        print(f"Found checkpoint {checkpoint}, {artifact_name}")

    artifact_name_sanitized: str = sanitize_fname(artifact.name.replace(":", "-"))
    print(f"download model {artifact_name_sanitized = }")

    download_dir: Path = output_path / "temp" / artifact_name_sanitized

    artifact.download(root=download_dir)
    # get the single .zanj file in the download dir
    download_path = next(download_dir.glob("*.zanj"))
    print(f"\tDownloaded model to '{download_path}'")

    print(f"load and re-save model with better filename and more data")
    print(f"\tLoading model from '{download_path}'")
    model: ZanjHookedTransformer = ZanjHookedTransformer.read(download_path)

    # add metadata to model training records
    model.training_records.update(
        dict(
            run_id=run_id,
            project=project,
            checkpoint=checkpoint,
            is_final=is_final,
            original_download_path=download_path,
            version=artifact.name.split(":")[-1],
        )
    )

    # save as proper model name
    updated_save_path: Path = output_path / (
        f"model.{artifact_name_sanitized}.iter_{checkpoint}.zanj"
        if not is_final
        else f"model.{artifact_name_sanitized}.final.zanj"
    )
    model.save(updated_save_path)

    print(f"\tSaved model to '{updated_save_path}'")

    return model
