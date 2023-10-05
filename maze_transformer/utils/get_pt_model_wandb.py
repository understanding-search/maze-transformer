import math
from pathlib import Path

import torch
import wandb
from maze_dataset import MazeDatasetConfig
from muutils.misc import sanitize_fname, shorten_numerical_to_str
from transformer_lens import HookedTransformer
from wandb.sdk.wandb_run import Artifact, Run

from maze_transformer.training.config import (
    BaseGPTConfig,
    ConfigHolder,
    TrainConfig,
    ZanjHookedTransformer,
)


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
            "are_layernorms_folded": True,
            "are_weights_processed": True,
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


def load_wandb_pt_model_as_zanj(
    run_id: str,
    project: str = "aisc-search/alex",
    checkpoint: int | None = None,
    output_path: str = "./downloaded_models",
    save_zanj_model: bool = True,
    verbose: bool = True,
) -> ZanjHookedTransformer:
    model_kwargs: dict = dict(
        project=project,
        run_id=run_id,
        checkpoint=checkpoint,
    )
    model: HookedTransformer
    cfg: ConfigHolder
    model, cfg = load_wandb_run(**model_kwargs)
    if verbose:
        print(f"{type(model) = } {type(cfg) = }")

    model_zanj: ZanjHookedTransformer = ZanjHookedTransformer(cfg)
    model_zanj.load_state_dict(model.state_dict())
    model_zanj.training_records = {
        "load_wandb_run_kwargs": model_kwargs,
        "train_cfg.name": cfg.train_cfg.name,
    }
    if verbose:
        print(
            f"loaded model with {shorten_numerical_to_str(model_zanj.num_params())} parameters"
        )
        print(model_zanj.training_records)

    if save_zanj_model:
        model_zanj_save_path: Path = (
            Path(output_path) / f"wandb.{model_kwargs['run_id']}.zanj"
        )
        model_zanj.save(model_zanj_save_path)
        if verbose:
            print(f"Saved model to {model_zanj_save_path.as_posix()}")

    return model_zanj


def load_wandb_zanj(
    run_id: str,
    project: str = "aisc-search/alex",
    checkpoint: int | None = None,
    output_path: str | Path = "./downloaded_models",
    model: bool = True,
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
