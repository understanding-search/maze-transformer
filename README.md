# Maze Transformer

Solving mazes with transformer models.

# Scripts

## `test_generation`
Generate a maze and solve it algorithmically.

### Example
```
python scripts/test_generation.py
```

## `create_dataset`
Create or load a dataset of mazes.

### Example
create 10 4x4 mazes in the directory ./data/maze:
```
python scripts/create_dataset.py create ./data/maze 10 --grid_n=4
```

## `train_model`
Uses a dataset to train a model. Outputs model model and configs in a subdirectory of the dataset directory.

Notes:
* This script DOES NOT require a GPU to run. If training a small model, you can run it on your laptop's CPU. The quickest way to set the device is to add `device="cpu"` to the `"tiny-v1"` TrainConfig, defined in `_TRAINING_CONFIG_LIST` in the file `config.py`.
* This script requires a dataset, which can be generated using `create_dataset.py`.
* The `"tiny-v1"` model is not well optimised (i.e. it won't make good predictions).

### Example
```
python scripts/train-model.py ./data/maze/g4-n10
```


# Development

## Setup

* Install [PyTorch](https://pytorch.org/get-started/locally/) to your global python
* Create venv and allow it to access global packages

    ```python -m venv venv_path --system-site-packages```
* Activate venv

    ```source venv_path/bin/activate```
* Install dependencies

    ```pip install -r requirements.txt```

* Install muutils in editable mode

    TODO: release update to muutils to remove this step
    ```
    mkdir -p /path/to/muutils/
    pushd /path/to/muutils/
    git clone git@github.com:mivanit/muutils .
    # This commit is known to be working
    git checkout 4a9dae2d

    popd
    pip install -e /path/to/muutils
    ````

* Install current package in editable mode

    This allows pytest to resolve the project packages.
    ```
    pip install -e .
    ```

* (Optional) Run unit tests

    ```
    make test
    ```


## Testing & Static analysis

- unit tests via `make unit`

- integration tests via `make integration`

- all tests via `make test`

- formatter (black, pycln, and isort) via `make format`

- formatter in check-only mode via `make check-format`
