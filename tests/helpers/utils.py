# Used for comparison of adjlists
# Adjlist looks like [[[0, 1], [1, 1]], [[0, 0], [0, 1]], ...]
# We don't care about order of coordinate pairs within
# the adjlist or coordinates within each coordinate pair.
def adjlist_to_nested_set(adjlist):
    return {
        frozenset([tuple(start_coord), tuple(end_coord)])
        for start_coord, end_coord in adjlist
    }
