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
python scripts/train_model.py ./data/maze/g4-n10
```


# Development

## Prerequisites
**We are using Python 3.10 (or newer) and Poetry.**
* Install [Poetry](https://python-poetry.org/docs/#installation)
* Install Python 3.10

## Setup

* Install dependencies
    ```
    poetry install
    ```


* (Optional) Run unit and integration tests
    ```
    make test
    ```


## Testing & Static analysis

- unit tests via `make unit`

- integration tests via `make integration`

- all tests via `make test`

- formatter (black, pycln, and isort) via `make format`

- formatter in check-only mode via `make check-format`
