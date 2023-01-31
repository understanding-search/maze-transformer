# Maze Transformer

Solving mazes with transformer models.

# Scripts

## `test_generation`
Generate a maze and solve it algorithmically.

```python test_generation.py [width] [height]```

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

    ```
    pushd /path/to/muutils/
    git clone git@github.com:mivanit/muutils

    popd
    pip install -e /path/to/muutils
    ````



## Testing & Static analysis

- formatter (black and isort) via `make format`
