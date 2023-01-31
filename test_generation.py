import time

import numpy as np

from maze_transformer.evaluation.plot_maze import plot_path
from maze_transformer.generation.generators import LatticeMazeGenerators
from maze_transformer.generation.latticemaze import LatticeMaze


def generate_solve_plot(shape_x: int = 5, shape_y: int | None = None):
    if shape_y is None:
        shape_y = shape_x

    t: float = time.time()
    m: LatticeMaze = LatticeMazeGenerators.gen_dfs(np.array([shape_x, shape_y]))
    print(f"generation time: {time.time() - t}")

    # show a path
    c_start = (0, 0)
    c_end = (shape_x - 1, shape_y - 1)

    t = time.time()
    path = np.array(
        m.find_shortest_path(
            c_start=c_start,
            c_end=c_end,
        )
    )

    print(f"solving time: {time.time() - t}")

    plot_path(m, path, show=True)


if __name__ == "__main__":
    import fire

    fire.Fire(generate_solve_plot)
