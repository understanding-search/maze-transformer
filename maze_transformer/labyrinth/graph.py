"""Object-oriented representation of the mathematical concept of a graph."""

from collections import defaultdict, deque
from typing import Callable, Collection, Generic, Optional, Set, Tuple, TypeVar


T = TypeVar('T')


class Graph(Generic[T]):
    """Class representing a graph."""

    def __init__(self, vertices: Optional[Collection[T]] = None, edges: Optional[Collection[Tuple[T, T]]] = None,
                 bidirectional: bool = True) -> None:
        """Initialize a Graph, optionally pre-populated with the given vertices and edges."""
        self._adjacencies = {}
        self._bidirectional = bidirectional

        if vertices is not None:
            for vertex in vertices:
                self.add_vertex(vertex)

        if edges is not None:
            for edge in edges:
                self.add_edge(*edge)

    @property
    def bidirectional(self) -> bool:
        """Return a boolean indicating whether this graph is bidirectional."""
        return self._bidirectional

    @property
    def vertices(self) -> Set[T]:
        """Return a set of all vertices in this graph."""
        return set(self._adjacencies.keys())

    @property
    def edges(self) -> Set[Tuple[T, T]]:
        edges = set()
        for vertex, neighbors in self._adjacencies.items():
            for neighbor in neighbors:
                if not self.bidirectional or (neighbor, vertex) not in edges:
                    edges.add((vertex, neighbor))
        return edges

    @property
    def size(self) -> int:
        """Return the number of vertices in this graph."""
        return len(self._adjacencies)

    def neighbors(self, vertex: T) -> Set[T]:
        """Return a set of all neighbors of the given vertex."""
        self._ensure_vertices(vertex)
        return self._adjacencies[vertex]

    def add_vertex(self, vertex: T) -> None:
        """Add the given vertex to this graph."""
        if vertex not in self._adjacencies:
            self._adjacencies[vertex] = set()

    def remove_vertex(self, vertex: T) -> None:
        """Remove the given vertex from this graph, as well as all edges connected to it."""
        self._ensure_vertices(vertex)
        neighbors = self._adjacencies[vertex]
        # removing a vertex implies removing the edges connected to that vertex
        for neighbor in neighbors:
            self._adjacencies[neighbor].remove(vertex)
        del self._adjacencies[vertex]

    def add_edge(self, left: T, right: T) -> None:
        """Add an edge between the given vertices to this graph."""
        self._ensure_vertices(left, right)
        self._adjacencies[left].add(right)
        if self._bidirectional:
            self._adjacencies[right].add(left)

    def remove_edge(self, left: T, right: T) -> None:
        """Remove the edge between the given vertices from this graph."""
        self._ensure_vertices(left, right)
        self._adjacencies[left].remove(right)
        if self._bidirectional:
            self._adjacencies[right].remove(left)

    def has_edge(self, left: T, right: T) -> bool:
        """Return a boolean indicating whether an edge exists between the given vertices in this graph."""
        self._ensure_vertices(left, right)
        return right in self._adjacencies[left]

    def breadth_first_search(self, start_vertex: T, visit_fn: Callable[[T], None] = print) -> None:
        """Perform a breadth-first search (BFS) of the graph, starting from the given vertex."""
        self._ensure_vertices(start_vertex)
        visited = defaultdict(bool)
        queue = deque([start_vertex])
        while queue:
            vertex = queue.popleft()
            if not visited[vertex]:
                visit_fn(vertex)
                visited[vertex] = True
                for neighbor in self.neighbors(vertex):
                    if not visited[neighbor]:
                        queue.append(neighbor)

    def depth_first_search(self, start_vertex: T, visit_fn: Callable[[T], None] = print) -> None:
        """Perform a depth-first search (DFS) of the graph, starting from the given vertex."""
        self._ensure_vertices(start_vertex)
        visited = defaultdict(bool)
        stack = [start_vertex]
        while stack:
            vertex = stack.pop()
            if not visited[vertex]:
                visit_fn(vertex)
                visited[vertex] = True
                for neighbor in self.neighbors(vertex):
                    if not visited[neighbor]:
                        stack.append(neighbor)

    def _ensure_vertices(self, *vertices: T) -> None:
        for vertex in vertices:
            if vertex not in self._adjacencies:
                raise ValueError(f'Invalid vertex {vertex!r}')