"""
Test that loading the model and configuration works

* I.e. the outputs of the model are identical when loading into
    a HookedTransformer with folding etc., as they would be from
    just applying the model to the input
"""
from pathlib import Path

import pytest
import torch

from muutils.zanj import ZANJ
from muutils.zanj.torchutil import (
    ConfigMismatchException,
    assert_model_cfg_equality,
    assert_model_exact_equality,
)

from maze_transformer.evaluation.eval_model import (
    evaluate_model,
    load_model_with_configs,
    predict_maze_paths,
)
from maze_transformer.evaluation.path_evals import PathEvals
from maze_transformer.training.config import ConfigHolder, ZanjHookedTransformer
from maze_transformer.training.maze_dataset import MazeDataset, MazeDatasetConfig
from maze_transformer.training.training import TRAIN_SAVE_FILES
from maze_transformer.training.wandb_logger import WandbProject
from scripts.create_dataset import create_dataset
from scripts.train_model import train_model, TrainingResult
from maze_transformer.evaluation.util import assert_model_output_equality

temp_dir: Path = Path("tests/_temp/test_eval_model")

def test_model_loading():
    # get config
    cfg: ConfigHolder = ConfigHolder.get_config_multisource(
        cfg_names=("test-g3-n5-a_dfs", "nano-v1", "integration-v1"),
    )
    # train model
    result: TrainingResult = train_model(
        base_path=temp_dir,
        wandb_project=WandbProject.INTEGRATION_TESTS,
        cfg=cfg,
        do_generate_dataset=True,
    )
    model_ret: ZanjHookedTransformer = result.model

    # load model
    model_load_auto: ZanjHookedTransformer = ZANJ().read(result.output_path / TRAIN_SAVE_FILES.model_final_zanj)

    # Load model manually without folding
    assert cfg == model_ret.zanj_model_config
    assert_model_cfg_equality(model_ret, model_load_auto)

    assert_model_output_equality(model_ret, model_load_auto, cfg)
    assert_model_exact_equality(model_ret, model_load_auto)


def test_predict_maze_paths():
    # Setup will be refactored in https://github.com/orgs/AISC-understanding-search/projects/1?pane=issue&itemId=22504590
    # First create a dataset and train a model
    # get config
    cfg: ConfigHolder = ConfigHolder.get_config_multisource(
        cfg_names=("test-g3-n5-a_dfs", "nano-v1", "integration-v1"),
    )
    # train model
    model_ret: ZanjHookedTransformer = train_model(
        base_path=temp_dir,
        wandb_project=WandbProject.INTEGRATION_TESTS,
        cfg=cfg,
        do_generate_dataset=True,
    )

    # Now load the model and compare the outputs
    # Get directory of the training run
    base_path = Path(temp_dir / "g3-n5-test")
    run_path = [x for x in base_path.glob("*") if x.is_dir()][0]

    # Load model using our function (with layernorm folding etc.)
    model, cfg = load_model_with_configs(run_path / "model.final.pt")

    dataset = MazeDataset.disk_load(path_base=base_path, do_config=True, do_tokens=True)

    max_new_tokens = 2
    paths = predict_maze_paths(
        tokens_batch=dataset.mazes_tokens,
        data_cfg=cfg.dataset_cfg,
        model=model,
        max_new_tokens=max_new_tokens,
    )

    all_coordinates = [coord for path in paths for coords in path for coord in coords]
    assert len(paths) == 5
    assert max([len(path) for path in paths]) <= max_new_tokens + 1
    assert max(all_coordinates) == grid_n - 1


@pytest.mark.usefixtures("temp_dir")
def test_evaluate_model(temp_dir):
    # Setup will be refactored in https://github.com/orgs/AISC-understanding-search/projects/1?pane=issue&itemId=22504590
    # First create a dataset and train a model
    # get config
    cfg: ConfigHolder = ConfigHolder.get_config_multisource(
        cfg_names=("test-g3-n5-a_dfs", "nano-v1", "integration-v1"),
    )
    # train model
    model_ret: ZanjHookedTransformer = train_model(
        base_path=temp_dir,
        wandb_project=WandbProject.INTEGRATION_TESTS,
        cfg=cfg,
        do_generate_dataset=True,
    )

    # Now load the model and compare the outputs
    # Get directory of the training run
    base_path = Path(temp_dir / "g3-n5-test")
    run_path = [x for x in base_path.glob("*") if x.is_dir()][0]

    # Load model using our function (with layernorm folding etc.)
    model, cfg = load_model_with_configs(run_path / "model.final.pt")

    dataset = MazeDataset.disk_load(path_base=base_path, do_config=True, do_tokens=True)

    path_evals = PathEvals.evals
    eval_names = [name for name in path_evals.keys()]
    scores = evaluate_model(dataset=dataset, model=model)

    assert path_evals.keys() == scores.keys()
    assert scores[eval_names[0]].summary()["total_items"] == n_mazes
