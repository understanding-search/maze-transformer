"""Classes for creating and working with grids of cells."""

from enum import Enum
from typing import Optional, Set, Tuple

from labyrinth.graph import Graph


class Cell:
    """Class representing a cell in a grid."""

    def __init__(self, row: int, column: int) -> None:
        """Initialize a Cell."""
        self._row = row
        self._column = column
        self.open_walls = set()

    def __repr__(self) -> str:
        """Return a string representation of the cell."""
        return f'{self.__class__.__name__}{self.coordinates}'

    def __str__(self) -> str:
        """Return a string representation of the cell."""
        return repr(self)

    @property
    def row(self) -> int:
        """Return the cell's row number."""
        return self._row

    @property
    def column(self) -> int:
        """Return the cell's column number."""
        return self._column

    @property
    def coordinates(self) -> Tuple[int, int]:
        """Return the cell's row and column as a two-tuple."""
        return self.row, self.column


class Direction(Enum):
    """Enumeration of the directions allowed for movement within a grid."""
    N = (0, -1)
    S = (0, 1)
    E = (1, 0)
    W = (-1, 0)

    @property
    def dx(self) -> int:
        """Return the change in x (column) when moving in this direction."""
        return self.value[0]

    @property
    def dy(self) -> int:
        """Return the change in y (row) when moving in this direction."""
        return self.value[1]

    @property
    def opposite(self) -> 'Direction':
        """Return the direction opposite to this direction."""
        return next(d for d in self.__class__ if (d.dx, d.dy) == (-self.dx, -self.dy))

    @classmethod
    def between(cls, start_cell: Cell, end_cell: Cell) -> Optional['Direction']:
        """Return the direction between the given start and end cells, which are assumed to be adjacent."""
        dx = end_cell.column - start_cell.column
        dy = end_cell.row - start_cell.row
        return next((d for d in cls if (d.dx, d.dy) == (dx, dy)), None)


class Grid:
    """Class representing a grid of cells as a graph."""

    def __init__(self, width: int = 10, height: int = 10) -> None:
        """Initialize a Grid."""
        self._width = width
        self._height = height
        self._cells = {}
        self._graph = Graph()
        for row in range(height):
            for column in range(width):
                coordinates = (row, column)
                cell = Cell(*coordinates)
                self._cells[coordinates] = cell
                self._graph.add_vertex(cell)
                if row > 0:
                    self._graph.add_edge(cell, self[row - 1, column])
                if column > 0:
                    self._graph.add_edge(cell, self[row, column - 1])

    def __getitem__(self, item: Tuple[int, int]) -> Cell:
        """Return the cell in the grid at the given coordinates."""
        return self.get_cell(*item)

    @property
    def width(self) -> int:
        """Return the width (number of columns) of the grid."""
        return self._width

    @property
    def height(self) -> int:
        """Return the height (number of rows) of the grid."""
        return self._height

    @property
    def graph(self) -> Graph[Cell]:
        """Return the graph representation underlying this grid."""
        return self._graph

    def get_cell(self, row: int, column: int) -> Cell:
        """Return the cell in the grid at the given row and column."""
        if not 0 <= row < self.height:
            raise ValueError(f'Invalid row {row!r}')
        if not 0 <= column < self.width:
            raise ValueError(f'Invalid column {column!r}')
        return self._cells[(row, column)]

    def neighbors(self, cell: Cell) -> Set[Cell]:
        """Return a set of all neighbors of the given cell in the grid."""
        return self._graph.neighbors(cell)