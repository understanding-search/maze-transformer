from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from muutils.tensor_utils import NDArray

from maze_transformer.generation.latticemaze import LatticeMaze

CoordArray = NDArray["coords", np.int8]

# Set size of unit.
# Each "unit" consists of a node and the right and lower adjacent wall/connection. Its area is ul * ul.
# Wall thickness is fixed to 1, so this parameter sets the ratio of node length to wall thickness
ul = 14


# Maze conversion functions (No plt involved)


def as_img(maze: LatticeMaze) -> NDArray["x y", bool]:
    """
     Build an image to visualise the maze.
     Each "unit" consists of a node and the right and lower adjacent wall/connection. Its area is ul * ul.
     - Nodes are displayed as white squares of area: (ul-1) * (ul-1)
     - Walls are displayed as black recktangles of area: 1 * (ul-1)
     - Connections are displayed as light grey or white rectangles of area: 1 * (ul-1); color is depending on show_connections argument

     Axes definition:
    (0,0)     y
      -|----------->
     x |
       |
       |
       v

     Returns a matrix of side length (ul) * n + 1 where n is the number of nodes.
    """

    # Set color of connections (using plt.imshow(cmap='grey))
    show_connections = True

    white_color = 100
    if show_connections:
        connection_color = white_color * 0.93
    else:
        connection_color = white_color

    # Set up the background (walls everywhere)
    img: NDArray["x y", int] = np.zeros(
        (
            maze.grid_shape[0] * ul + 1,
            maze.grid_shape[1] * ul + 1,
        ),
        dtype=int,
    )

    # Draw nodes and connections by iterating through lattice
    for x in range(maze.grid_shape[0]):
        for y in range(maze.grid_shape[1]):
            # Draw node
            img[x * ul + 1 : (x + 1) * ul, y * ul + 1 : (y + 1) * ul] = white_color

            # Down connection
            if maze.connection_list[0, x, y]:
                img[(x + 1) * ul, y * ul + 1 : (y + 1) * ul] = connection_color

            # Right connection
            if maze.connection_list[1, x, y]:
                img[x * ul + 1 : (x + 1) * ul, (y + 1) * ul] = connection_color

    return img


def points_transform_to_img(points: CoordArray) -> CoordArray:
    """transform points to img coordinates"""

    #####
    # Swapping x, y axes here may be deprecated with resolution of issue #73
    points = np.array([(x, y) for (y, x) in points])
    #####

    return ul * (points + 0.5)


# Image rendering functions (plt involved)


def plot_maze(maze: LatticeMaze, show: bool = True) -> None:
    """ul : ratio of wall thickness and node length"""

    img = as_img(maze)
    plt.imshow(img, cmap="gray")

    # Plot labels
    tick_arr = np.arange(maze.grid_shape[0])
    plt.xticks(ul * (tick_arr + 0.5), tick_arr)
    plt.yticks(ul * (tick_arr + 0.5), tick_arr)
    plt.xlabel("y")
    plt.ylabel("x")
    if show:
        plt.show()


def plot_path(maze: LatticeMaze, path: NDArray, show: bool = True) -> None:
    # print(m)
    # show the maze
    path_transformed = points_transform_to_img(path)

    # dimension swap -  we are translating from array notation to cartesian coordinates
    plt.plot(path_transformed[:, 1], path_transformed[:, 0], "-", color="red")
    # mark endpoints
    plt.plot([path_transformed[0][1]], [path_transformed[0][0]], "o", color="red")
    plt.plot([path_transformed[-1][1]], [path_transformed[-1][0]], "x", color="red")
    # show actual maze
    plot_maze(maze, show=False)

    if show:
        plt.show()


@dataclass
class PathFormat:
    """formatting options for path plot
    (path_true, "true", "-", "red")
    """

    path: NDArray
    label: str | None = None
    fmt: str | None = None
    color: str | None = None
    quiver_kwargs: dict | None = (None,)


def plot_multi_paths(
    maze: LatticeMaze,
    paths: list[PathFormat | tuple | list],
    show: bool = True,
) -> None:
    # plot paths
    for pf in paths:
        if isinstance(pf, (tuple, list)):
            pf = PathFormat(*pf)

        p_transformed = points_transform_to_img(np.array(pf.path))
        if pf.quiver_kwargs is not None:
            # Pyplot uses Cartesian coordinates (x horixontal and y vertical)
            # But our mazes and paths use array notation (row, col)
            # So we swap the dimensions here
            x: NDArray = p_transformed[:, 1]
            y: NDArray = p_transformed[:, 0]
            plt.quiver(
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
            plt.plot(*zip(*p_transformed), pf.fmt, color=pf.color, label=pf.label)
        # mark endpoints
        plt.plot([p_transformed[0][1]], [p_transformed[0][0]], "o", color=pf.color)
        plt.plot([p_transformed[-1][1]], [p_transformed[-1][0]], "x", color=pf.color)

    # show actual maze
    plot_maze(maze, show=False)

    if show:
        plt.show()
