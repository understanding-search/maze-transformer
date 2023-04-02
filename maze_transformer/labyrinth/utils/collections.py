"""Utility collection classes."""


class DisjointSet:
    """
    DisjointSet is an implementation of a disjoint-set data structure.
    References:
        https://en.wikipedia.org/wiki/Disjoint-set_data_structure
        https://weblog.jamisbuck.org/2011/1/3/maze-generation-kruskal-s-algorithm
    """

    def __init__(self) -> None:
        """Initialize a DisjointSet."""
        self.parent = None

    @property
    def root(self) -> 'DisjointSet':
        """Return the root of this set."""
        if self.parent is None:
            return self
        return self.parent.root

    def is_connected(self, tree: 'DisjointSet') -> bool:
        """Return True if this set is connected to the given set, False otherwise."""
        return self.root == tree.root

    def merge(self, tree: 'DisjointSet') -> None:
        """Merge the given set into this set."""
        tree.root.parent = self