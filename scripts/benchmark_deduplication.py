from maze_transformer.dataset.maze_dataset import MazeDataset, MazeDatasetConfig
import time


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
