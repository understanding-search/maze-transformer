"""Generate mazes using a variety of different algorithms."""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Set, Tuple
import abc
import random

from labyrinth.grid import Cell, Direction
from labyrinth.utils.collections import DisjointSet
from labyrinth.utils.abc import override
from labyrinth.utils.event import EventDispatcher


class MazeUpdateType(Enum):
    """Enumeration of all maze update event types."""
    START_CELL_CHOSEN = 1
    CELL_MARKED = 2
    EDGE_REMOVED = 3
    WALL_REMOVED = 4


@dataclass
class MazeUpdate:
    """Data class holding the state of an update to a maze."""
    type: MazeUpdateType
    start_cell: Cell
    end_cell: Optional[Cell] = None
    new_frontier_cells: Optional[Set[Cell]] = None


class MazeGenerator(abc.ABC, EventDispatcher[MazeUpdate]):
    """Abstract base class for a maze generator."""

    def __init__(self, event_listener: Optional[Callable[[MazeUpdate], None]] = None) -> None:
        """Initialize a MazeGenerator with an optional event listener."""
        super().__init__(event_listener=event_listener)
        self.maze = None

    def open_wall(self, start_cell: Cell, end_cell: Cell) -> None:
        """Open the wall between the given cells in the maze and invoke the event listener (if any)."""
        if self.maze is None:
            raise ValueError('No current maze to remove a wall from!')
        self.maze.open_wall(start_cell, end_cell)
        state = MazeUpdate(type=MazeUpdateType.WALL_REMOVED, start_cell=start_cell, end_cell=end_cell)
        super().on_state_changed(state)

    def get_random_cell(self) -> Cell:
        """Return a random cell in the maze."""
        if self.maze is None:
            raise ValueError('No current maze to get a random cell from!')
        return self.maze[random.randrange(self.maze.height), random.randrange(self.maze.width)]

    def get_random_start_cell(self) -> Cell:
        """Return a random cell in the maze and notify any event listener that the chosen cell is the starting cell."""
        cell = self.get_random_cell()
        state = MazeUpdate(type=MazeUpdateType.START_CELL_CHOSEN, start_cell=cell)
        super().on_state_changed(state)
        return cell

    def generate(self, maze: 'Maze') -> None:
        """Generate paths through the given maze."""
        self.maze = maze
        self.generate_maze()
        self.maze = None

    @abc.abstractmethod
    def generate_maze(self) -> None:
        """Generate paths through a maze."""
        raise NotImplemented


class DepthFirstSearchGenerator(MazeGenerator):
    """
    MazeGenerator subclass that generates mazes using the depth-first search (DFS) algorithm.
    This algorithm is also known as the 'recursive backtrack' algorithm. The algorithm is equivalent to the following:
    def recursive_backtrack(maze, row=0, column=0):
        cell = maze[row, column]
        for neighbor in maze.neighbors(cell):
            if not neighbor.open_walls:
                maze.open_wall(cell, neighbor)
                recursive_backtrack(maze, neighbor.row, neighbor.column)
    """

    def __init__(self, event_listener: Optional[Callable[[MazeUpdate], None]] = None) -> None:
        """Initialize a DepthFirstSearchGenerator with an optional event listener."""
        super().__init__(event_listener)
        self.prev_cells = None

    def cell_visitor(self, cell: Cell) -> None:
        """Visitor function for the depth-first search algorithm."""
        if cell in self.prev_cells:
            self.open_wall(self.prev_cells[cell], cell)
        for neighbor in self.maze.neighbors(cell):
            if not neighbor.open_walls:
                self.prev_cells[neighbor] = cell

    @override
    def generate_maze(self) -> None:
        """Generate paths through a maze using random depth-first search."""
        self.prev_cells = {}
        self.maze.depth_first_search(self.maze.start_cell, self.cell_visitor)


class KruskalsGenerator(MazeGenerator):
    """MazeGenerator subclass that generates mazes using a modified version of Kruskal's algorithm."""

    def __init__(self, event_listener: Optional[Callable[[MazeUpdate], None]] = None) -> None:
        """Initialize a KruskalsGenerator with an optional event listener."""
        super().__init__(event_listener)

    def on_edge_removed(self, start_cell: Cell, end_cell: Cell) -> None:
        """Notify any event listeners when an edge is removed from the graph."""
        state = MazeUpdate(type=MazeUpdateType.EDGE_REMOVED, start_cell=start_cell, end_cell=end_cell)
        self.on_state_changed(state)

    @override
    def generate_maze(self) -> None:
        """Generate paths through a maze using a modified version of Kruskal's algorithm."""
        walls = self.maze.walls
        sets = defaultdict(DisjointSet)
        while walls:
            start_cell, end_cell = walls.pop()
            start_set, end_set = sets[start_cell], sets[end_cell]
            if start_set.is_connected(end_set):
                self.on_edge_removed(start_cell, end_cell)
            else:
                start_set.merge(end_set)
                self.open_wall(start_cell, end_cell)


class PrimsGenerator(MazeGenerator):
    """MazeGenerator subclass that generates mazes using a modified version of Prim's algorithm."""

    def __init__(self, event_listener: Optional[Callable[[MazeUpdate], None]] = None) -> None:
        """Initialize a PrimsGenerator with an optional event listener."""
        super().__init__(event_listener)
        self.included_cells = None
        self.frontier = None

    def on_cell_marked(self, cell: Cell, new_frontier_cells: Set[Cell]):
        """Notify any event listeners when a cell is marked as included in the maze."""
        state = MazeUpdate(type=MazeUpdateType.CELL_MARKED, start_cell=cell, new_frontier_cells=new_frontier_cells)
        self.on_state_changed(state)

    def mark(self, cell: Cell) -> None:
        """Mark a cell as being part of the maze, and add its neighbors to the set of frontier cells."""
        self.included_cells.add(cell)
        new_frontier_cells = set(n for n in self.maze.neighbors(cell) if not n.open_walls)
        self.frontier |= new_frontier_cells
        self.on_cell_marked(cell, new_frontier_cells)

    @override
    def generate_maze(self) -> None:
        """Generate paths through a maze using a modified version of Prim's algorithm."""
        self.included_cells = set()
        self.frontier = set()
        start_cell = self.get_random_start_cell()
        self.mark(start_cell)
        while self.frontier:
            next_cell = self.frontier.pop()
            neighbor = random.choice([c for c in self.maze.neighbors(next_cell) if c in self.included_cells])
            self.open_wall(next_cell, neighbor)
            self.mark(next_cell)


class WilsonsGenerator(MazeGenerator):
    """MazeGenerator subclass that generates mazes using Wilson's algorithm."""

    def __init__(self, event_listener: Optional[Callable[[MazeUpdate], None]] = None) -> None:
        """Initialize a WilsonsGenerator with an optional event listener."""
        super().__init__(event_listener)
        self.included_cells = None

    def get_random_unvisited_cell(self) -> Cell:
        """Return a random cell in the maze that has not been visited yet."""
        cell = None
        while cell is None or cell in self.included_cells:
            cell = self.get_random_cell()
        return cell

    def walk(self) -> List[Tuple[Cell, Direction]]:
        """Perform a random walk through unvisited cells of the maze and return the path walked."""
        start_cell = self.get_random_unvisited_cell()
        visits = {}
        cell = start_cell

        while True:
            neighbor = random.choice(list(self.maze.neighbors(cell)))  # pick a random neighbor
            direction = Direction.between(cell, neighbor)
            visits[cell] = direction
            if neighbor in self.included_cells:
                break
            cell = neighbor

        path = []
        cell = start_cell
        while cell in visits:
            direction = visits[cell]
            path.append((cell, direction))
            cell = self.maze.neighbor(cell, direction)
        return path

    @override
    def generate_maze(self) -> None:
        """Generate paths through a maze using Wilson's algorithm for generating uniform spanning trees."""
        self.included_cells = set()
        start_cell = self.get_random_start_cell()
        self.included_cells.add(start_cell)
        while len(self.included_cells) < self.maze.width * self.maze.height:
            for cell, direction in self.walk():
                neighbor = self.maze.neighbor(cell, direction)
                self.open_wall(cell, neighbor)
                self.included_cells.add(cell)
