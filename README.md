# Maze Transformer

Solving mazes with transformer models.

# Scripts

## `test_generation`
Generate a maze and solve it algorithmically.

### Example
```
python3 test_generation.py
```

## `create_dataset`
Create or load a dataset of mazes.

### Example
create 10 4x4 mazes in the directory ./data/maze:
```
python3 create_dataset.py create ./data/maze 10 --grid_n=4
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

    **TODO: the package@commit seems to have been added to requirements.txt, so this step may not be necessary. Need to test.**

    ```
    pushd /path/to/muutils/
    git clone git@github.com:mivanit/muutils

    popd
    pip install -e /path/to/muutils
    ````



## Testing & Static analysis

- formatter (black and isort) via `make format`
