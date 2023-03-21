from pathlib import Path

import wandb


def upload_dataset(name: str, path: Path):
    wandb.init(project="understanding-search", job_type="create-dataset")
    artifact = wandb.Artifact(name=name, type="dataset")
    artifact.add_dir(local_path=path)
    wandb.log_artifact(artifact)


if __name__ == "__main__":
    import fire

    fire.Fire(upload_dataset)
