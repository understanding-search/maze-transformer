from __future__ import annotations  # for type hinting self as return value

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from muutils.tensor_utils import NDArray

from maze_transformer.generation.latticemaze import CoordArray, LatticeMaze


@dataclass
class PathFormat:
    """formatting options for path plot
    (path_true, "true", "-", "red")
    """

    path: CoordArray
    label: str | None = None
    fmt: str | None = None
    color: str | None = None
    line_width: float | None = None
    quiver_kwargs: dict | None = (None,)


class MazePlot:
    """Class for displaying mazes and paths

    UNIT_LENGTH: Set ratio between node size and wall thickness in image.
    Wall thickness is fixed to 1px
    A "unit" consists of a single node and the right and lower connection/wall.
    Example: ul = 14 yields 13:1 ratio between node size and wall thickness
    """

    UNIT_LENGTH = 14
    DEFAULT_PREDICTED_PATH_COLORS = [
        "tab:blue",
        "tab:green",
        "tab:purple",
        "tab:orange",
        "tab:olive",
        "tab:cyan",
    ]

    def __init__(self, maze: LatticeMaze) -> None:
        self.maze: LatticeMaze = maze
        self.true_path: PathFormat = None
        self.predicted_paths: list = []

    def add_true_path(
        self,
        path: CoordArray,
        label: str = "true path",
        fmt: str = "--",
        color: str = "red",
        line_width: float = 2.5,
        quiver_kwargs: dict | None = None,
    ) -> MazePlot:
        """
        Recieve true path and formatting preferences from input and save as dict in true_path variable.
        """
        self.true_path = PathFormat(
            path=path,
            label=label,
            fmt=fmt,
            color=color,
            line_width=line_width,
            quiver_kwargs=quiver_kwargs,
        )
        return self

    def add_predicted_path(
        self,
        path: CoordArray,
        label: str | None = None,
        fmt: str = ":",
        color: str | None = None,
        line_width: float = 2,
        quiver_kwargs: dict | None = {"width": 0.015},
    ) -> MazePlot:
        """
        Recieve predicted path and formatting preferences from input and save in predicted_path list.
        Default formatting depends on nuber of paths already saved in predicted path list.
        """
        if label is None:
            label = "predicted path {path_num}".format(
                path_num=len(self.predicted_paths) + 1
            )
        if color is None:
            color = self.DEFAULT_PREDICTED_PATH_COLORS[len(self.predicted_paths)]

        self.predicted_paths.append(
            PathFormat(
                path=path,
                label=label,
                fmt=fmt,
                color=color,
                line_width=line_width,
                quiver_kwargs=quiver_kwargs,
            )
        )
        return self

    def _latticemaze_to_img(
        self, unit_length: int = UNIT_LENGTH
    ) -> NDArray["row col", bool]:
        """
        Build an image to visualise the maze.
        Each "unit" consists of a node and the right and lower adjacent wall/connection. Its area is ul * ul.
        - Nodes are displayed as white squares of area: (ul-1) * (ul-1)
        - Walls are displayed as black recktangles of area: 1 * (ul-1)
        - Connections are displayed as light grey or white rectangles of area: 1 * (ul-1); color is depending on show_connections argument

        Axes definition:
        (0,0)     y, col
        ----|----------->
        x,  |
        row |
            |
            v

        Returns a matrix of side length (ul) * n + 1 where n is the number of nodes.
        """
        # Set color of connections (using plt.imshow(cmap='grey))
        node_color = 1
        connection_color = node_color * 0.93

        # Set up the background (walls everywhere)
        img: NDArray["row col", int] = np.zeros(
            (
                self.maze.grid_shape[0] * unit_length + 1,
                self.maze.grid_shape[1] * unit_length + 1,
            ),
            dtype=float,
        )

        # Draw nodes and connections by iterating through lattice
        for row in range(self.maze.grid_shape[0]):
            for col in range(self.maze.grid_shape[1]):
                # Draw node
                img[
                    row * unit_length + 1 : (row + 1) * unit_length,
                    col * unit_length + 1 : (col + 1) * unit_length,
                ] = node_color

                # Down connection
                if self.maze.connection_list[0, row, col]:
                    img[
                        (row + 1) * unit_length,
                        col * unit_length + 1 : (col + 1) * unit_length,
                    ] = connection_color

                # Right connection
                if self.maze.connection_list[1, row, col]:
                    img[
                        row * unit_length + 1 : (row + 1) * unit_length,
                        (col + 1) * unit_length,
                    ] = connection_color

        return img

    def _rowcol_to_coord(self, points: CoordArray) -> NDArray:
        """transform points to img coordinates"""
        points = np.array([(x, y) for (y, x) in points])
        return self.UNIT_LENGTH * (points + 0.5)

    def as_ascii(self, start=None, end=None):
        """
        Returns an ASCII visualization of the maze.
        Courtesy of ChatGPT
        """
        wall_char = "#"
        path_char = " "

        # Determine the size of the maze
        maze = self._latticemaze_to_img(unit_length=2)
        n_rows, n_cols = maze.shape
        maze_str = ""

        # Iterate through each element of the maze and print the appropriate symbol
        for i in range(n_rows):
            for j in range(n_cols):
                if start is not None and start[0] == i - 1 and start[1] == j - 1:
                    maze_str += "S"
                elif end is not None and end[0] == i - 1 and end[1] == j - 1:
                    maze_str += "E"
                elif maze[i, j]:
                    maze_str += path_char
                else:
                    maze_str += wall_char
            maze_str += "\n"  # Start a new line after each row
        return maze_str

    # Image rendering functions (plt involved)
    def _plot_maze(self) -> None:
        img = self._latticemaze_to_img()
        self.ax.imshow(img, cmap="gray", vmin=0, vmax=1)

    def _plot_path(self, pf: PathFormat) -> None:
        p_transformed = self._rowcol_to_coord(pf.path)
        if pf.quiver_kwargs is not None:
            x: NDArray = p_transformed[:, 0]
            y: NDArray = p_transformed[:, 1]
            self.ax.quiver(
                x[:-1],
                y[:-1],
                x[1:] - x[:-1],
                y[1:] - y[:-1],
                scale_units="xy",
                angles="xy",
                scale=1,
                color=pf.color,
                **pf.quiver_kwargs,
            )
        else:
            self.ax.plot(
                *zip(*p_transformed),
                pf.fmt,
                lw=pf.line_width,
                color=pf.color,
                label=pf.label,
            )
        # mark endpoints
        self.ax.plot([p_transformed[0][1]], [p_transformed[0][0]], "o", color=pf.color)
        self.ax.plot(
            [p_transformed[-1][1]], [p_transformed[-1][0]], "x", color=pf.color
        )

    def show(self) -> None:
        """Plot the maze and paths."""
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

        self._plot_maze()

        if self.true_path is not None:
            self._plot_path(self.true_path)
        for path in self.predicted_paths:
            self._plot_path(path)

        # Plot labels
        tick_arr = np.arange(self.maze.grid_shape[0])
        self.ax.set_xticks(self.UNIT_LENGTH * (tick_arr + 0.5), tick_arr)
        self.ax.set_yticks(self.UNIT_LENGTH * (tick_arr + 0.5), tick_arr)
        self.ax.set_xlabel("col")
        self.ax.set_ylabel("row")

        plt.show()
