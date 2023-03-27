"""
Test that loading the model and configuration works

* I.e. the outputs of the model are identical when loading into
    a HookedTransformer with folding etc., as they would be from
    just applying the model to the input
"""
from pathlib import Path

import pytest
import torch

from maze_transformer.evaluation.eval_model import (
    evaluate_model,
    load_model_with_configs,
    predict_maze_paths,
)
from maze_transformer.evaluation.path_evals import PathEvals
from maze_transformer.training.mazedataset import MazeDataset
from maze_transformer.training.wandb_logger import WandbProject
from scripts.create_dataset import create_dataset
from scripts.train_model import train_model


@pytest.mark.usefixtures("temp_dir")
def test_model_loading(temp_dir):
    # First create a dataset and train a model
    #! Awaiting change of all paths to Path for training scripts
    if not Path.exists(temp_dir / "g3-n5-test"):
        create_dataset(path_base=str(temp_dir), n_mazes=5, grid_n=3, name="test")
        train_model(
            basepath=str(temp_dir / "g3-n5-test"),
            wandb_project=WandbProject.INTEGRATION_TESTS,
            training_cfg="integration-v1",
            model_cfg="nano-v1",
        )

    # Now load the model and compare the outputs
    # Get directory of the training run
    run_folder_path = Path(temp_dir / "g3-n5-test")
    run_folder_path = [x for x in run_folder_path.glob("*") if x.is_dir()][0]

    # Load model using our function (with layernorm folding etc.)
    model, cfg = load_model_with_configs(run_folder_path / "model.final.pt")

    # Load model manually without folding
    model_state_dict = torch.load(run_folder_path / "model.final.pt")
    model_basic = cfg.create_model()
    model_basic.load_state_dict(model_state_dict)

    # Random input tokens
    input_sequence = torch.randint(
        low=0,
        high=len(cfg.dataset_cfg.token_arr),
        size=(1, min(cfg.dataset_cfg.seq_len_max, 10)),
    )

    # Check for equality in argsort (absolute values won't be equal due to centering the unembedding weight matrix)
    # Alternatively could apply normalization (e.g. softmax) and check with atol v-small
    # (roughly 1E-7 for float error on logexp I think)
    assert torch.all(
        model(input_sequence.clone()).argsort()
        == model_basic(input_sequence.clone()).argsort()
    )


@pytest.mark.usefixtures("temp_dir")
def test_predict_maze_paths(temp_dir):
    # Setup will be refactored in https://github.com/orgs/AISC-understanding-search/projects/1?pane=issue&itemId=22504590
    # First create a dataset and train a model
    grid_n = 3
    if not Path.exists(temp_dir / "g3-n5-test"):
        create_dataset(path_base=str(temp_dir), n_mazes=5, grid_n=grid_n, name="test")
        train_model(
            basepath=str(temp_dir / "g3-n5-test"),
            training_cfg="integration-v1",
            model_cfg="nano-v1",
            wandb_project=WandbProject.INTEGRATION_TESTS,
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
    n_mazes = 5
    if not Path.exists(temp_dir / "g3-n5-test"):
        create_dataset(path_base=str(temp_dir), n_mazes=n_mazes, grid_n=3, name="test")
        train_model(
            basepath=str(temp_dir / "g3-n5-test"),
            training_cfg="integration-v1",
            model_cfg="nano-v1",
            wandb_project=WandbProject.INTEGRATION_TESTS,
        )

    # Now load the model and compare the outputs
    # Get directory of the training run
    base_path = Path(temp_dir / "g3-n5-test")
    run_path = [x for x in base_path.glob("*") if x.is_dir()][0]

    # Load model using our function (with layernorm folding etc.)
    model, cfg = load_model_with_configs(run_path / "model.final.pt")

    dataset = MazeDataset.disk_load(path_base=base_path, do_config=True, do_tokens=True)

    path_evals = PathEvals.all_functions()
    eval_names = [name for name in path_evals.keys()]
    scores = evaluate_model(dataset=dataset, model=model)

    assert path_evals.keys() == scores.keys()
    assert scores[eval_names[0]].summary()["total_items"] == n_mazes
