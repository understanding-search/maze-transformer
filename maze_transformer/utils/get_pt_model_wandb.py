import math
from pathlib import Path

import torch
import wandb
from maze_dataset import MazeDatasetConfig
from muutils.misc import shorten_numerical_to_str
from transformer_lens import HookedTransformer
from wandb.sdk.wandb_run import Artifact, Run

from maze_transformer.training.config import (
    BaseGPTConfig,
    ConfigHolder,
    TrainConfig,
    ZanjHookedTransformer,
)


def get_step(artifact: Artifact) -> int:
    # Find the alias beginning with "step="
    step_alias: list[str] = [
        alias for alias in artifact.aliases if alias.startswith("step=")
    ]
    if len(step_alias) != 1:  # if we have multiple, skip as well
        return -1
    return int(step_alias[0].split("=")[-1])


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


def load_wandb_run(
    project="aisc-search/alex",
    run_id="sa973hyn",
    output_path="./downloaded_models",
    checkpoint=None,
) -> tuple[HookedTransformer, ConfigHolder]:
    api: wandb.Api = wandb.Api()

    artifact_name: str = f"{project.rstrip('/')}/{run_id}"

    run: Run = api.run(artifact_name)
    wandb_cfg: wandb.config.Config = run.config  # Get run configuration

    # -- Get / Match checkpoint --
    if checkpoint is not None:
        # Match checkpoint
        available_checkpoints = [
            artifact for artifact in run.logged_artifacts() if artifact.type == "model"
        ]
        available_checkpoints = list(run.logged_artifacts())
        artifact = [aft for aft in available_checkpoints if get_step(aft) == checkpoint]
        if len(artifact) != 1:
            print(f"Could not find checkpoint {checkpoint} in {artifact_name}")
            print("Available checkpoints:")
            [
                print(artifact.name, "| Steps: ", get_step(artifact))
                for artifact in available_checkpoints
            ]
            return

        artifact = artifact[0]
        print("Loading checkpoint", checkpoint)
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
