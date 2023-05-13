import random
import warnings
from typing import Any, Callable

import numpy as np

from maze_transformer.generation.constants import CoordArray
from maze_transformer.generation.lattice_maze import (
    NEIGHBORS_MASK,
    ConnectionList,
    Coord,
    LatticeMaze,
    SolvedMaze,
)


class LatticeMazeGenerators:
    """namespace for lattice maze generation algorithms"""

    @staticmethod
    def gen_dfs(
        grid_shape: Coord,
        lattice_dim: int = 2,
        n_accessible_cells: int | None = None,
        max_tree_depth: int | None = None,
        start_coord: Coord | None = None,
    ) -> LatticeMaze:
        """generate a lattice maze using depth first search, iterative

        algorithm:
        1. Choose the initial cell, mark it as visited and push it to the stack
        2. While the stack is not empty
                1. Pop a cell from the stack and make it a current cell
                2. If the current cell has any neighbours which have not been visited
                        1. Push the current cell to the stack
                        2. Choose one of the unvisited neighbours
                        3. Remove the wall between the current cell and the chosen cell
                        4. Mark the chosen cell as visited and push it to the stack
        """

        # Default values if no constraints have been passed
        grid_shape: Coord = np.array(grid_shape)
        n_total_cells: int = int(np.prod(grid_shape))
        if n_accessible_cells is None:
            n_accessible_cells = n_total_cells
        if max_tree_depth is None:
            max_tree_depth = (
                2 * n_total_cells
            )  # We define max tree depth counting from the start coord in two directions. Therefore we divide by two in the if clause for neighboring sites later and multiply by two here.
        if start_coord is None:
            start_coord: Coord = np.random.randint(
                0,
                np.maximum(grid_shape - 1, 1),
                size=2,
            )
        else:
            start_coord = np.array(start_coord)

        # initialize the maze with no connections
        connection_list: ConnectionList = np.zeros(
            (lattice_dim, grid_shape[0], grid_shape[1]), dtype=np.bool_
        )

        # initialize the stack with the target coord
        visited_cells: set[tuple[int, int]] = set()
        visited_cells.add(tuple(start_coord))
        stack: list[Coord] = [start_coord]

        # initialize tree_depth_counter
        current_tree_depth: int = 1

        # loop until the stack is empty or n_connected_cells is reached
        while stack and (len(visited_cells) < n_accessible_cells):
            # get the current coord from the stack
            current_coord: Coord = stack.pop()

            # filter neighbors by being within grid bounds and being unvisited
            unvisited_neighbors_deltas: list[tuple[Coord, Coord]] = [
                (neighbor, delta)
                for neighbor, delta in zip(
                    current_coord + NEIGHBORS_MASK, NEIGHBORS_MASK
                )
                if (
                    (tuple(neighbor) not in visited_cells)
                    and (0 <= neighbor[0] < grid_shape[0])
                    and (0 <= neighbor[1] < grid_shape[1])
                )
            ]

            # don't continue if max_tree_depth/2 is already reached (divide by 2 because we can branch to multiple directions)
            if unvisited_neighbors_deltas and (
                current_tree_depth <= max_tree_depth / 2
            ):
                stack.append(current_coord)

                # choose one of the unvisited neighbors
                chosen_neighbor, delta = random.choice(unvisited_neighbors_deltas)

                # add connection
                dim: int = np.argmax(np.abs(delta))
                # if positive, down/right from current coord
                # if negative, up/left from current coord (down/right from neighbor)
                clist_node: Coord = (
                    current_coord if (delta.sum() > 0) else chosen_neighbor
                )
                connection_list[dim, clist_node[0], clist_node[1]] = True

                # add to visited cells and stack
                visited_cells.add(tuple(chosen_neighbor))
                stack.append(chosen_neighbor)

                # Update current tree depth
                current_tree_depth += 1
            else:
                current_tree_depth -= 1

        return LatticeMaze(
            connection_list=connection_list,
            generation_meta=dict(
                func_name="gen_dfs",
                grid_shape=grid_shape,
                start_coord=start_coord,
                visited_cells=visited_cells,
                n_accessible_cells=n_accessible_cells,
                max_tree_depth=max_tree_depth,
                fully_connected=(len(visited_cells) == n_accessible_cells),
            ),
        )

    @staticmethod
    def gen_wilson(
        grid_shape: Coord,
    ) -> LatticeMaze:
        """Generate a lattice maze using Wilson's algorithm. Wilson's algorithm generates an unbiased (random) maze
        sampled from the uniform distribution over all mazes, using loop-erased random walks. The generated maze is
        acyclic and all cells are part of a unique connected space.
        https://en.wikipedia.org/wiki/Maze_generation_algorithm#Wilson's_algorithm"""

        def neighbor(current: Coord, direction: int) -> Coord:
            row, col = current

            if direction == 0:
                col -= 1  # Left
            elif direction == 1:
                col += 1  # Right
            elif direction == 2:
                row -= 1  # Up
            elif direction == 3:
                row += 1  # Down
            else:
                return None

            return np.array([row, col]) if 0 <= row < rows and 0 <= col < cols else None

        rows, cols = grid_shape

        # A connection list only contains two elements: one boolean matrix indicating all the
        # downwards connections in the maze, and one boolean matrix indicating the rightwards connections.
        connection_list: np.ndarray = np.zeros((2, rows, cols), dtype=np.bool_)

        connected = np.zeros(grid_shape, dtype=np.bool_)
        direction_matrix = np.zeros(grid_shape, dtype=int)

        # Mark a random cell as connected
        connected[random.randint(0, rows - 1)][random.randint(0, cols - 1)] = True

        cells_left: int = rows * cols - 1
        while cells_left > 0:
            current: Coord = np.array(
                [random.randint(0, rows - 1), random.randint(0, cols - 1)]
            )
            start: Coord = current

            # Random walk through the maze while recording path taken until a connected cell is found
            while not connected[current[0]][current[1]]:
                # Find a valid neighboring cell by checking in a random direction then rotating clockwise
                direction: int = random.randint(0, 4)
                next: Coord = neighbor(current, direction)

                while next is None:
                    direction += 1
                    if direction > 3:
                        direction = 0
                    next = neighbor(current, direction)

                # Keep track of the random path
                direction_matrix[current[0]][current[1]] = direction
                # Move to the neighboring cell
                current = next

            direction_matrix[current[0]][current[1]] = 4

            # Return to the start and retrace our path, connecting cells as we go
            current = start
            while not connected[current[0]][current[1]]:
                direction = direction_matrix[current[0]][current[1]]
                connected[current[0]][current[1]] = True
                cells_left -= 1

                next = neighbor(current, direction)
                # Connect the current and next cell
                # todo(luciaq) update LatticeMaze to take an adjacency list instead of a connection list for a more
                # natural connection update here
                if direction == 0:  # Left
                    connection_list[1][next[0]][next[1]] = True
                elif direction == 1:  # Right
                    connection_list[1][current[0]][current[1]] = True
                elif direction == 2:  # Up
                    connection_list[0][next[0]][next[1]] = True
                elif direction == 3:  # Down
                    connection_list[0][current[0]][current[1]] = True

                current = next

        return LatticeMaze(
            connection_list=connection_list,
            generation_meta=dict(
                func_name="gen_wilson",
                grid_shape=grid_shape,
                fully_connected=True,
            ),
        )

    @classmethod
    def gen_dfs_with_solution(cls, grid_shape: Coord):
        warnings.warn(
            "gen_dfs_with_solution is deprecated, use get_maze_with_solution instead",
            DeprecationWarning,
        )
        return get_maze_with_solution("gen_dfs", grid_shape)


# TODO: use the thing @valedan wrote for the evals function to make this automatic?
GENERATORS_MAP: dict[str, Callable[[Coord, Any], "LatticeMaze"]] = {
    "gen_dfs": LatticeMazeGenerators.gen_dfs,
    "gen_wilson": LatticeMazeGenerators.gen_wilson,
}


def get_maze_with_solution(
    gen_name: str,
    grid_shape: Coord,
    maze_ctor_kwargs: dict | None = None,
) -> SolvedMaze:
    if maze_ctor_kwargs is None:
        maze_ctor_kwargs = dict()
    maze: LatticeMaze = GENERATORS_MAP[gen_name](grid_shape, **maze_ctor_kwargs)
    solution: CoordArray = np.array(maze.generate_random_path())
    return SolvedMaze.from_lattice_maze(lattice_maze=maze, solution=solution)
