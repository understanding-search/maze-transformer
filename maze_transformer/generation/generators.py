import random
from typing import Any, Callable

import numpy as np
from generation.utils import neighbor

from maze_transformer.generation.latticemaze import (
    NEIGHBORS_MASK,
    Coord,
    LatticeMaze,
    SolvedMaze,
)


class LatticeMazeGenerators:
    """namespace for lattice maze generation algorithms"""

    @staticmethod
    def gen_dfs(
        grid_shape: Coord,
        start_coord: Coord | None = None,
        lattice_dim: int = 2,
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

        # n_directions: int = lattice_dim * 2

        # initialize the maze with no connections
        connection_list: np.ndarray = np.zeros(
            (lattice_dim, grid_shape[0], grid_shape[1]), dtype=bool
        )

        if start_coord is None:
            start_coord: Coord = (
                random.randint(0, grid_shape[0] - 1),
                random.randint(0, grid_shape[1] - 1),
            )

        # print(f"{grid_shape = } {start_coord = }")

        # initialize the stack with the target coord
        visited_cells: set[tuple[int, int]] = set()
        visited_cells.add(tuple(start_coord))
        stack: list[Coord] = [start_coord]

        # loop until the stack is empty
        while stack:
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

            if unvisited_neighbors_deltas:
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

        return LatticeMaze(
            connection_list=connection_list,
            generation_meta=dict(
                func_name="gen_dfs",
                grid_shape=grid_shape,
                start_coord=start_coord,
            ),
        )

    @classmethod
    def gen_dfs_with_solution(cls, grid_shape: Coord):
        maze = cls.gen_dfs(grid_shape)
        solution = np.array(maze.generate_random_path())

        return SolvedMaze(maze, solution)

    @staticmethod
    def gen_wilson(
            grid_shape: Coord,
    ) -> LatticeMaze:
        """generate a lattice maze using Wilson's algorithm
        https://en.wikipedia.org/wiki/Maze_generation_algorithm#Wilson's_algorithm"""

        height, width = grid_shape

        # Contrary to its name, a connection list only ever contains two elements: one boolean matrix indicating all the
        # downwards connections in the maze, and one boolean matrix for all the rightwards connections.
        connection_list: np.ndarray = np.zeros(
            (2, height, width), dtype=bool
        )

        connected = np.zeros(grid_shape, dtype=bool)
        direction_matrix = np.zeros(grid_shape)
        connected[0][0] = True


        nodes_left: int = height * width
        while nodes_left > 0:
            current: Coord = connection_list[:][random.randint(0, height - 1)][random.randint(0, width - 1)]
            start: Coord = current

            # random walk through the maze until a connected cell is found
            while not connected[current[0]][current[1]]:
                # find a valid neighboring cell by checking in a random direction then rotating clockwise
                direction: int = random.randint(0, 4)
                next_cell: Coord = neighbor(current, direction, connection_list)

                while next_cell is None:
                    direction += 1
                    if direction > 3:
                        direction = 0
                    next_cell = neighbor(current, direction, connection_list)

                # keep track of the random path
                direction_matrix[current[0]][current[1]] = direction
                # move to the neighboring cell
                current = next_cell

            direction_matrix[current[0]][current[1]] = 4

            # Backtrack and retrace our path, connecting cells as we go
            current = start
            while not connected[current[0]][current[1]]:
                direction = direction_matrix[current[0]][current[1]]
                connected[current[0]][current[1]] = True
                nodes_left -= 1

                next_cell = neighbor(current, direction, connection_list)
                # add edge
                connection_list[direction][current[0]][current[1]] = True

                current = next_cell

        return LatticeMaze(
            connection_list=connection_list,
            generation_meta=dict(
                func_name="gen_wilson",
                grid_shape=grid_shape,
            ),
        )



GENERATORS_MAP: dict[str, Callable[[Coord, Any], "LatticeMaze"]] = {
    "gen_dfs": LatticeMazeGenerators.gen_dfs,
    "gen_wilson": LatticeMazeGenerators.gen_wilson
}
