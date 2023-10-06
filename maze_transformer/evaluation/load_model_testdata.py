from pathlib import Path

from maze_dataset import MazeDataset, MazeDatasetConfig
from muutils.misc import shorten_numerical_to_str

from maze_transformer.training.config import ZanjHookedTransformer


def load_model_with_test_data(
    model_path: str | Path,
    dataset_cfg_source: MazeDatasetConfig | None = None,
    n_examples: int | None = 100,
    verbose: bool = True,
) -> tuple[ZanjHookedTransformer, MazeDataset]:
    model_path = Path(model_path)

    # load model
    model: ZanjHookedTransformer = ZanjHookedTransformer.read(model_path)
    num_params: int = model.num_params()
    model_name: str = str(model_path.stem).removeprefix("model.").removeprefix("wandb.")

    if verbose:
        print(
            f"loaded model with {shorten_numerical_to_str(num_params)} params ({num_params = }) from\n{model_path.as_posix()}"
        )
        print(
            f"original model name: '{model.zanj_model_config.name = }', changing to '{model_name}'"
        )

    model.zanj_model_config.name = model_name

    # copy config if needed, adjust number of mazes
    if dataset_cfg_source is None:
        # deep copy of config
        dataset_cfg_source = MazeDatasetConfig.load(
            model.zanj_model_config.dataset_cfg.serialize()
        )

    # adjust number of mazes
    if n_examples is not None:
        dataset_cfg_source.n_mazes = n_examples

    # get the dataset
    dataset: MazeDataset = MazeDataset.from_config(dataset_cfg_source)

    if verbose:
        print(f"loaded dataset with {len(dataset)} examples")
        print(f"{dataset.cfg.summary() = }")

    return model, dataset
