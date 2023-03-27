from pathlib import Path
import json

from muutils.zanj import ZANJ
from maze_transformer.training.config import ConfigHolder, MazeDatasetConfig, TrainConfig, BaseGPTConfig


def test_misc():
	dataset_cfg=MazeDatasetConfig(name="test_cfg_save-data", grid_n=5, n_mazes=10)
	print(dataset_cfg)
	print(dataset_cfg.serialize())

	assert dataset_cfg == MazeDatasetConfig.load(dataset_cfg.serialize())


def test_cfg_save():

	fname: Path = Path("tests/_temp/test_cfg_save.json")
	fname.parent.mkdir(parents=True, exist_ok=True)

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

	with open(fname, "w") as f:
		json.dump(cfg.serialize(), f, indent="\t")

	with open(fname, "r") as f:
		loaded = ConfigHolder.load(json.load(f))

	print(loaded)

	assert loaded == cfg, f"{loaded} != {cfg}"

def test_cfg_save_zanj():
	fname: Path = Path("tests/_temp/test_cfg_save_z.zanj")
	fname.parent.mkdir(parents=True, exist_ok=True)

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

	zanj = ZANJ()

	zanj.save(cfg, fname)

	loaded = zanj.read(fname)

	assert loaded == cfg, f"{loaded} != {cfg}"