"""Classes for creating and working with mazes."""

from typing import Callable, Optional, Set, Tuple

from labyrinth.generate import DepthFirstSearchGenerator, MazeGenerator
from labyrinth.grid import Cell, Direction, Grid


class Maze:
    """Representation of a maze as a graph with a grid structure."""

    def __init__(self, width: int = 10, height: int = 10,
                 generator: Optional[MazeGenerator] = DepthFirstSearchGenerator()) -> None:
        """Initialize a Maze."""
        self._grid = Grid(width, height)
        self.path = []
        if generator:
            generator.generate(self)

    def __getitem__(self, item: Tuple[int, int]) -> Cell:
        """Return the cell in the maze at the given coordinates."""
        return self._grid[item]

    def __str__(self) -> str:
        """Return a string representation of the maze."""
        cell_width = 3
        maze_str = '+' + ((('-' * cell_width) + '+') * self.width) + '\n'
        for row in range(self.height):
            maze_str += '|'
            for column in range(self.width):
                cell = self[row, column]
                maze_str += ' X ' if cell in self.path else ' ' * cell_width
                maze_str += ' ' if Direction.E in cell.open_walls else '|'
            maze_str += '\n+'
            for column in range(self.width):
                maze_str += (' ' if Direction.S in self[row, column].open_walls else '-') * cell_width
                maze_str += '+'
            maze_str += '\n'
        return maze_str

    @property
    def width(self) -> int:
        """Return the width (number of columns) of this maze."""
        return self._grid.width

    @property
    def height(self) -> int:
        """Return the height (number of rows) of this maze."""
        return self._grid.height

    @property
    def walls(self) -> Set[Tuple[Cell, Cell]]:
        """Return a set of all walls in this maze."""
        return self._grid.graph.edges

    @property
    def start_cell(self) -> Cell:
        """Return the starting cell of this maze."""
        return self[0, 0]

    @property
    def end_cell(self) -> Cell:
        """Return the ending cell of this maze."""
        return self[self.height - 1, self.width - 1]

    def get_cell(self, row: int, column: int) -> Cell:
        """Return the cell in the maze at the given row and column."""
        return self._grid.get_cell(row, column)

    def neighbors(self, cell: Cell) -> Set[Cell]:
        """Return a set of all neighbors of the given cell in the maze."""
        return self._grid.neighbors(cell)

    def neighbor(self, cell: Cell, direction: Direction) -> Optional[Cell]:
        """Return the cell neighboring the given cell in the given direction, if there is one."""
        return next((n for n in self.neighbors(cell) if Direction.between(cell, n) == direction), None)

    def depth_first_search(self, start_cell: Cell, visit_fn: Callable[[Cell], None]):
        """Perform a depth-first search of the maze, starting from the given cell."""
        self._grid.graph.depth_first_search(start_cell, visit_fn)

    @staticmethod
    def open_wall(start_cell: Cell, end_cell: Cell) -> None:
        """Open (remove) the walls between the given start and end cells, which are assumed to be adjacent."""
        direction = Direction.between(start_cell, end_cell)
        start_cell.open_walls.add(direction)
        end_cell.open_walls.add(direction.opposite)