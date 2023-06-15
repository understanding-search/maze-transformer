import time

from maze_dataset import MazeDataset, MazeDatasetConfig

dataset = MazeDataset.generate(
    cfg=MazeDatasetConfig(
        name="test",
        grid_n=6,
        n_mazes=100000,
    )
)

start = time.time()
deduplicated = dataset.filter_by.remove_duplicates_fast()
end = time.time()

print(f"Time taken: {end - start}")
print(f"Original: {len(dataset)}")
print(f"Deduplicated: {len(deduplicated)}")
