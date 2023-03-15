import time

import numpy as np

from maze_transformer.evaluation.plot_maze import plot_path
from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.generation.latticemaze import LatticeMaze


# start/end usage: --start=0,0 --end=4,4
def generate_solve_plot(
    width: int = 5, height: int | None = None, start=None, end=None
):
    if height is None:
        height = width

    generation_start: float = time.time()
    maze: LatticeMaze = LatticeMazeGenerators.gen_dfs(np.array([width, height]))
    print(f"generation time: {time.time() - generation_start}")

    # show a path

    solution_start = time.time()

    if start and end:
        path = np.array(maze.find_shortest_path(start, end))
    else:
        path = np.array(maze.generate_random_path())

    print(f"solving time: {time.time() - solution_start}")

    plot_path(maze, path, show=True)


if __name__ == "__main__":
    import fire

    fire.Fire(generate_solve_plot)
