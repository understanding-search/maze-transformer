from typing import Callable
from maze_transformer.training.maze_dataset import MazeDatasetConfig, MazeDataset
from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.training.wandb_logger import (
    WandbJobType,
    WandbLogger,
    WandbProject,
)
from maze_transformer.training.config import BaseGPTConfig, ConfigHolder, GPT_CONFIGS, TRAINING_CONFIGS, ZanjHookedTransformer
from maze_transformer.training.training import get_dataloader, train
import datetime

def _create_dataset(n_mazes: int = 1000, grid_n: int = 8, maze_ctor: Callable = LatticeMazeGenerators.gen_dfs, do_parallel: bool = True, verbose: bool = True) -> tuple[MazeDataset, MazeDatasetConfig]:
    start_time = datetime.datetime.now()
    dataset_config = MazeDatasetConfig(name="benchmark", grid_n=grid_n, n_mazes=n_mazes)
    dataset_config.maze_ctor = maze_ctor
    print("Generating dataset...")
    dataset = MazeDataset.generate(dataset_config, do_parallel=do_parallel, verbose=verbose)
    end_time = datetime.datetime.now()
    print("dataset time: ", end_time - start_time)
    return dataset, dataset_config

def _create_config(dataset_config: MazeDatasetConfig, model: str = "tiny-v1", training_config: str = "integration-v1") -> ConfigHolder:
    config = ConfigHolder(name="benchmark", dataset_cfg=dataset_config, model_cfg=GPT_CONFIGS[model], train_cfg=TRAINING_CONFIGS[training_config])
    return config

def _create_logger(config: ConfigHolder) -> WandbLogger:
    logger: WandbLogger = WandbLogger.create(
        config=config.serialize(),
        project=WandbProject.INTEGRATION_TESTS,
        job_type=WandbJobType.TRAIN_MODEL,
    )
    return logger

def benchmark_dataset(n_mazes: int = 1000, grid_n: int = 8):
    print("Running dataset benchmarks")
    _create_dataset(n_mazes=n_mazes, grid_n=grid_n, maze_ctor=LatticeMazeGenerators.gen_dfs)

def benchmark_training(n_mazes: int = 1000, grid_n: int = 8):
    print('Running training benchmarks')
    dataset, dataset_config = _create_dataset(n_mazes=n_mazes, grid_n=grid_n, maze_ctor=LatticeMazeGenerators.gen_empty)
    config = _create_config(dataset_config=dataset_config)
    logger = _create_logger(config=config)

    print("Training model...")
    start_time = datetime.datetime.now()
    trained_model: ZanjHookedTransformer = train(dataset=dataset, cfg=config, logger=logger, verbose=True)
    end_time = datetime.datetime.now()

    print("Results:")
    print("training time: ", end_time - start_time)


def benchmark_dataloader(n_mazes: int = 1000, grid_n: int = 8):
    print("Running dataloader benchmarks")
    dataset, dataset_config = _create_dataset(n_mazes=n_mazes, grid_n=grid_n, maze_ctor=LatticeMazeGenerators.gen_empty)
    config = _create_config(dataset_config=dataset_config)
    logger = _create_logger(config=config)

    print("Creating dataloader...")
    start_time = datetime.datetime.now()
    dataloader = get_dataloader(dataset, config, logger)

    print("Dataloader created - iterating...")
    for i, batch in enumerate(dataloader):
        print("batch ", i, " of ", len(dataloader))

    end_time = datetime.datetime.now()
    print("Results:")
    print("dataloader time: ", end_time - start_time)

if __name__ == "__main__":
    import fire

    fire.Fire({
        "dataset": benchmark_dataset,
        "training": benchmark_training,
        "dataloader": benchmark_dataloader
    })
