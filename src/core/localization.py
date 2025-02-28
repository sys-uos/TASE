import networkx as nx

def calculate_weighted_centroid_from_nxgraph(graph: nx.DiGraph):
    """
    Calculates the weighted centroid of a directed graph based on node weights and positions.

    Each node should have:
    - 'weight': a numerical value
    - 'position': a tuple (x, y)

    Returns:
    - A tuple (centroid_x, centroid_y)
    """
    total_weight = 0
    weighted_sum_x = 0
    weighted_sum_y = 0

    for node, data in graph.nodes(data=True):
        weight = data.get('weight', 0)
        position = data.get('pos', (0, 0))

        if weight == 0:
            continue  # Skip nodes with zero weight

        x, y = position
        weighted_sum_x += weight * x
        weighted_sum_y += weight * y
        total_weight += weight

    centroid_x = weighted_sum_x / total_weight
    centroid_y = weighted_sum_y / total_weight

    return (centroid_x, centroid_y)