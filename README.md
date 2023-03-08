# Maze Transformer

Solving mazes with transformer models.

# Installation
```
pip install git+ssh://git@github.com/aisc-understanding-search/maze-transformer.git
```

Note: if you want to install the library in colab, follow the steps in this [Colab notebook](https://colab.research.google.com/drive/1b8E1rkqcKRdC4bs9133aBPEvqEaH5dqD#scrollTo=8VbjoPRgXlqs).

# Development

## Prerequisites

* Install [Poetry](https://python-poetry.org/docs/#installation)
* Install Python 3.10
    * It's a good idea to use [pyenv](https://github.com/pyenv/pyenv) to manage python versions
    * If using pyenv, you'll need to update your Poetry config for it to use the pyenv Python version: `poetry config virtualenvs.prefer-active-python true`


## Setup

* Install dependencies
    ```
    poetry install --with dev
    ```


* Run unit and integration tests
    ```
    make test
    ```

* (Optional) If you want to work with the jupyter notebooks in VSCode
  * create a jupyter kernel with `poetry run ipython kernel install --user --name=maze-transformer`
  * Restart VSCode
  * In VSCode, select the python interpreter located in `maze-transformer/.venv/bin` as your juptyer kernel

# Scripts

## `test_generation`
Generate a maze and solve it algorithmically.

### Example
```
poetry run python scripts/test_generation.py
```

## `create_dataset`
Create or load a dataset of mazes.

### Example
create 10 4x4 mazes in the directory ./data/maze:
```
poetry run python scripts/create_dataset.py create ./data/maze 10 --grid_n=4
```

## `train_model`
Uses a dataset to train a model. Outputs model model and configs in a subdirectory of the dataset directory.

Notes:
* This script DOES NOT require a GPU to run. If training a small model, you can run it on your laptop's CPU. The quickest way to set the device is to add `device="cpu"` to the `"tiny-v1"` TrainConfig, defined in `_TRAINING_CONFIG_LIST` in the file `config.py`.
* This script requires a dataset, which can be generated using `create_dataset.py`.
* The `"tiny-v1"` model is not well optimised (i.e. it won't make good predictions).

### Example
```
poetry run python scripts/train_model.py ./data/maze/g4-n10
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
