import json
from pathlib import Path

from maze_dataset import MazeDatasetConfig
from zanj import ZANJ

from maze_transformer.training.config import BaseGPTConfig, ConfigHolder, TrainConfig


def test_misc():
    dataset_cfg = MazeDatasetConfig(name="test_cfg_save-data", grid_n=5, n_mazes=10)
    print(dataset_cfg)
    print(dataset_cfg.serialize())

    assert dataset_cfg == MazeDatasetConfig.load(dataset_cfg.serialize())


cfg = ConfigHolder(
    train_cfg=TrainConfig(name="test_cfg_save-train"),
    dataset_cfg=MazeDatasetConfig(name="test_cfg_save-data", grid_n=5, n_mazes=10),
    model_cfg=BaseGPTConfig(
        name="test_cfg_save-model",
        act_fn="dummy-act-fn",
        d_model=500,
        d_head=60,
        n_layers=4,
    ),
)


def test_cfg_save():
    fname: Path = Path("tests/_temp/test_cfg_save.json")
    fname.parent.mkdir(parents=True, exist_ok=True)

    with open(fname, "w") as f:
        json.dump(cfg.serialize(), f, indent="\t")

    with open(fname, "r") as f:
        loaded = ConfigHolder.load(json.load(f))

    print(loaded)

    assert loaded == cfg, f"{loaded} != {cfg}"


def test_cfg_save_zanj():
    fname: Path = Path("tests/_temp/test_cfg_save_z.zanj")
    fname.parent.mkdir(parents=True, exist_ok=True)

    zanj = ZANJ()

    zanj.save(cfg, fname)

    loaded = zanj.read(fname)

    assert loaded == cfg, f"{loaded} != {cfg}"


cfg_pretrained_kwargs = ConfigHolder(
    train_cfg=TrainConfig(name="test_cfg_save-train"),
    dataset_cfg=MazeDatasetConfig(name="test_cfg_save-data", grid_n=5, n_mazes=10),
    model_cfg=BaseGPTConfig(
        name="test_cfg_save-model",
        act_fn="dummy-act-fn",
        d_model=500,
        d_head=60,
        n_layers=4,
    ),
    pretrainedtokenizer_kwargs=dict(
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    ),
)


def test_cfg_save_pretrained_tok():
    fname: Path = Path("tests/_temp/test_cfg_save_pretrained_tok.json")
    fname.parent.mkdir(parents=True, exist_ok=True)

    with open(fname, "w") as f:
        json.dump(cfg_pretrained_kwargs.serialize(), f, indent="\t")

    with open(fname, "r") as f:
        loaded = ConfigHolder.load(json.load(f))

    print(loaded)

    assert loaded == cfg_pretrained_kwargs, f"{loaded} != {cfg_pretrained_kwargs}"


def test_cfg_save_pretrained_tok_zanj():
    fname: Path = Path("tests/_temp/test_cfg_save_pretrained_tok_z.zanj")
    fname.parent.mkdir(parents=True, exist_ok=True)
    zanj = ZANJ()
    zanj.save(cfg_pretrained_kwargs, fname)
    loaded = zanj.read(fname)
    assert loaded == cfg_pretrained_kwargs, f"{loaded} != {cfg_pretrained_kwargs}"


def test_loading_example():
    data = {
        "__format__": "ConfigHolder(SerializableDataclass)",
        "name": "default",
        "train_cfg": {
            "__format__": "TrainConfig(SerializableDataclass)",
            "name": "test-v1",
            "optimizer": "RMSprop",
            "optimizer_kwargs": {"lr": 0.0001},
            "batch_size": 16,
            "dataloader_cfg": {
                "shuffle": True,
                "num_workers": 2,
                "persistent_workers": True,
                "drop_last": False,
            },
            "print_loss_interval": 100,
            "checkpoint_interval": 1000,
        },
        "dataset_cfg": {
            "__format__": "MazeDatasetConfig(SerializableDataclass)",
            "name": "g3-n5-test",
            "seq_len_min": 1,
            "seq_len_max": 512,
            "grid_n": 3,
            "n_mazes": 5,
            "maze_ctor": {
                "__name__": "gen_dfs",
                "__module__": "maze_transformer.generation.generators",
                "__doc__": ["dummy_doc"],
                "source_code": ["dummy_source_code"],
            },
            "padding_token_index": 10,
            "token_arr": [
                "<ADJLIST_START>",
                "<ADJLIST_END>",
                "<TARGET_START>",
                "<TARGET_END>",
                "<ORIGIN_START>",
                "<ORIGIN_END>",
                "<PATH_START>",
                "<PATH_END>",
                "<-->",
                ";",
                "<PADDING>",
                "(0,0)",
                "(0,1)",
                "(0,2)",
                "(1,0)",
                "(1,1)",
                "(1,2)",
                "(2,0)",
                "(2,1)",
                "(2,2)",
            ],
            "tokenizer_map": {
                "<ADJLIST_START>": 0,
                "<ADJLIST_END>": 1,
                "<TARGET_START>": 2,
                "<TARGET_END>": 3,
                "<ORIGIN_START>": 4,
                "<ORIGIN_END>": 5,
                "<PATH_START>": 6,
                "<PATH_END>": 7,
                "<-->": 8,
                ";": 9,
                "<PADDING>": 10,
                "(0,0)": 11,
                "(0,1)": 12,
                "(0,2)": 13,
                "(1,0)": 14,
                "(1,1)": 15,
                "(1,2)": 16,
                "(2,0)": 17,
                "(2,1)": 18,
                "(2,2)": 19,
            },
            "grid_shape": [3, 3],
            "token_node_map": {
                "(0,0)": [0, 0],
                "(0,1)": [0, 1],
                "(0,2)": [0, 2],
                "(1,0)": [1, 0],
                "(1,1)": [1, 1],
                "(1,2)": [1, 2],
                "(2,0)": [2, 0],
                "(2,1)": [2, 1],
                "(2,2)": [2, 2],
            },
            "n_tokens": 20,
        },
        "model_cfg": {
            "__format__": "BaseGPTConfig(SerializableDataclass)",
            "name": "nano-v1",
            "act_fn": "gelu",
            "d_model": 8,
            "d_head": 4,
            "n_layers": 2,
        },
    }

    cfg = ConfigHolder.load(data)

    assert isinstance(cfg, ConfigHolder)
    assert isinstance(cfg.train_cfg, TrainConfig)
    assert isinstance(cfg.dataset_cfg, MazeDatasetConfig)
    assert isinstance(cfg.model_cfg, BaseGPTConfig)


def test_loading_example_pretrained_tok():
    data2 = {
        "__format__": "ConfigHolder(SerializableDataclass)",
        "name": "default",
        "train_cfg": {
            "__format__": "TrainConfig(SerializableDataclass)",
            "name": "test-v1",
            "optimizer": "RMSprop",
            "optimizer_kwargs": {"lr": 0.0001},
            "batch_size": 16,
            "dataloader_cfg": {
                "shuffle": True,
                "num_workers": 2,
                "persistent_workers": True,
                "drop_last": False,
            },
            "print_loss_interval": 100,
            "checkpoint_interval": 1000,
        },
        "dataset_cfg": {
            "__format__": "MazeDatasetConfig(SerializableDataclass)",
            "name": "g3-n5-test",
            "seq_len_min": 1,
            "seq_len_max": 512,
            "grid_n": 3,
            "n_mazes": 5,
            "maze_ctor": {
                "__name__": "gen_dfs",
                "__module__": "maze_transformer.generation.generators",
                "__doc__": ["dummy_doc"],
                "source_code": ["dummy_source_code"],
            },
            "padding_token_index": 10,
            "token_arr": [
                "<ADJLIST_START>",
                "<ADJLIST_END>",
                "<TARGET_START>",
                "<TARGET_END>",
                "<ORIGIN_START>",
                "<ORIGIN_END>",
                "<PATH_START>",
                "<PATH_END>",
                "<-->",
                ";",
                "<PADDING>",
                "(0,0)",
                "(0,1)",
                "(0,2)",
                "(1,0)",
                "(1,1)",
                "(1,2)",
                "(2,0)",
                "(2,1)",
                "(2,2)",
            ],
            "tokenizer_map": {
                "<ADJLIST_START>": 0,
                "<ADJLIST_END>": 1,
                "<TARGET_START>": 2,
                "<TARGET_END>": 3,
                "<ORIGIN_START>": 4,
                "<ORIGIN_END>": 5,
                "<PATH_START>": 6,
                "<PATH_END>": 7,
                "<-->": 8,
                ";": 9,
                "<PADDING>": 10,
                "(0,0)": 11,
                "(0,1)": 12,
                "(0,2)": 13,
                "(1,0)": 14,
                "(1,1)": 15,
                "(1,2)": 16,
                "(2,0)": 17,
                "(2,1)": 18,
                "(2,2)": 19,
            },
            "grid_shape": [3, 3],
            "token_node_map": {
                "(0,0)": [0, 0],
                "(0,1)": [0, 1],
                "(0,2)": [0, 2],
                "(1,0)": [1, 0],
                "(1,1)": [1, 1],
                "(1,2)": [1, 2],
                "(2,0)": [2, 0],
                "(2,1)": [2, 1],
                "(2,2)": [2, 2],
            },
            "n_tokens": 20,
        },
        "model_cfg": {
            "__format__": "BaseGPTConfig(SerializableDataclass)",
            "name": "nano-v1",
            "act_fn": "gelu",
            "d_model": 8,
            "d_head": 4,
            "n_layers": 2,
        },
        "pretrainedtokenizer_kwargs": {
            "bos_token": "<PADDING>",
            "eos_token": "<PADDING>",
            "pad_token": "<PADDING>",
        },
    }

    cfg2 = ConfigHolder.load(data2)

    assert isinstance(cfg2, ConfigHolder)
    assert isinstance(cfg2.train_cfg, TrainConfig)
    assert isinstance(cfg2.dataset_cfg, MazeDatasetConfig)
    assert isinstance(cfg2.model_cfg, BaseGPTConfig)
    assert isinstance(cfg2.pretrainedtokenizer_kwargs, dict)


def test_other_example():
    data3 = {
        "__format__": "ConfigHolder(SerializableDataclass)",
        "name": "default",
        "train_cfg": {
            "__format__": "TrainConfig(SerializableDataclass)",
            "name": "test_cfg_save-train",
            "optimizer": "RMSprop",
            "optimizer_kwargs": {"lr": 1e-06},
            "batch_size": 128,
            "dataloader_cfg": {
                "shuffle": True,
                "num_workers": 16,
                "persistent_workers": True,
                "drop_last": True,
            },
            "print_loss_interval": 1000,
            "checkpoint_interval": 50000,
        },
        "dataset_cfg": {
            "__format__": "MazeDatasetConfig(SerializableDataclass)",
            "name": "test_cfg_save-data",
            "seq_len_min": 1,
            "seq_len_max": 512,
            "grid_n": 5,
            "n_mazes": 10,
            "maze_ctor": {
                "__name__": "gen_dfs",
                "__module__": "maze_transformer.generation.generators",
                "__doc__": ["dummy_doc"],
                "source_code": ["dummy_source_code"],
            },
            "padding_token_index": 10,
            "token_arr": [
                "<ADJLIST_START>",
                "<ADJLIST_END>",
                "<TARGET_START>",
                "<TARGET_END>",
                "<ORIGIN_START>",
                "<ORIGIN_END>",
                "<PATH_START>",
                "<PATH_END>",
                "<-->",
                ";",
                "<PADDING>",
                "(0,0)",
                "(0,1)",
                "(0,2)",
                "(0,3)",
                "(0,4)",
                "(1,0)",
                "(1,1)",
                "(1,2)",
                "(1,3)",
                "(1,4)",
                "(2,0)",
                "(2,1)",
                "(2,2)",
                "(2,3)",
                "(2,4)",
                "(3,0)",
                "(3,1)",
                "(3,2)",
                "(3,3)",
                "(3,4)",
                "(4,0)",
                "(4,1)",
                "(4,2)",
                "(4,3)",
                "(4,4)",
            ],
            "tokenizer_map": {
                "<ADJLIST_START>": 0,
                "<ADJLIST_END>": 1,
                "<TARGET_START>": 2,
                "<TARGET_END>": 3,
                "<ORIGIN_START>": 4,
                "<ORIGIN_END>": 5,
                "<PATH_START>": 6,
                "<PATH_END>": 7,
                "<-->": 8,
                ";": 9,
                "<PADDING>": 10,
                "(0,0)": 11,
                "(0,1)": 12,
                "(0,2)": 13,
                "(0,3)": 14,
                "(0,4)": 15,
                "(1,0)": 16,
                "(1,1)": 17,
                "(1,2)": 18,
                "(1,3)": 19,
                "(1,4)": 20,
                "(2,0)": 21,
                "(2,1)": 22,
                "(2,2)": 23,
                "(2,3)": 24,
                "(2,4)": 25,
                "(3,0)": 26,
                "(3,1)": 27,
                "(3,2)": 28,
                "(3,3)": 29,
                "(3,4)": 30,
                "(4,0)": 31,
                "(4,1)": 32,
                "(4,2)": 33,
                "(4,3)": 34,
                "(4,4)": 35,
            },
            "grid_shape": [5, 5],
            "token_node_map": {
                "(0,0)": [0, 0],
                "(0,1)": [0, 1],
                "(0,2)": [0, 2],
                "(0,3)": [0, 3],
                "(0,4)": [0, 4],
                "(1,0)": [1, 0],
                "(1,1)": [1, 1],
                "(1,2)": [1, 2],
                "(1,3)": [1, 3],
                "(1,4)": [1, 4],
                "(2,0)": [2, 0],
                "(2,1)": [2, 1],
                "(2,2)": [2, 2],
                "(2,3)": [2, 3],
                "(2,4)": [2, 4],
                "(3,0)": [3, 0],
                "(3,1)": [3, 1],
                "(3,2)": [3, 2],
                "(3,3)": [3, 3],
                "(3,4)": [3, 4],
                "(4,0)": [4, 0],
                "(4,1)": [4, 1],
                "(4,2)": [4, 2],
                "(4,3)": [4, 3],
                "(4,4)": [4, 4],
            },
            "n_tokens": 36,
        },
        "model_cfg": {
            "__format__": "BaseGPTConfig(SerializableDataclass)",
            "name": "test_cfg_save-model",
            "act_fn": "dummy-act-fn",
            "d_model": 500,
            "d_head": 60,
            "n_layers": 4,
        },
        "pretrainedtokenizer_kwargs": {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
        },
    }

    cfg3 = ConfigHolder.load(data3)

    assert isinstance(cfg3, ConfigHolder)
    assert isinstance(cfg3.train_cfg, TrainConfig)
    assert isinstance(cfg3.dataset_cfg, MazeDatasetConfig)
    assert isinstance(cfg3.model_cfg, BaseGPTConfig)
    assert isinstance(cfg3.pretrainedtokenizer_kwargs, dict)
