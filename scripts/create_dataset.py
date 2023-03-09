from maze_transformer.training.mazedataset import MazeDataset
from maze_transformer.generation.create import generate_MazeTokenizer, create_dataset


def test_load_dataset(path: str) -> None:
    """see if loading a dataset works"""
    d = MazeDataset.disk_load(path, do_tokenized=True)

    print(d.cfg)
    print(d.mazes_array)

    print("done!")


if __name__ == "__main__":
    import fire

    fire.Fire(
        dict(
            create=create_dataset,
            load=test_load_dataset,
        )
    )
