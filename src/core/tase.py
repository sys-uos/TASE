import os.path
from typing import Dict, Any, List

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from collections import deque

from TASE.src.models import Recording_Node
from TASE.src.utils import eucl_dist, convert_str_keys_to_int


class CustomizedGraph:

    def __init__(self):
        pass

    def __calculate_bounding_box(self) -> Dict[str, float]:
        """
        Calculate geographical bounding box for location data.

        Returns:
            Dict[str, float]: Bounding box with min/max latitude and longitude.
        """
        self.bounding_box = {
            "min_lat": min(node.lat for node in self.location_data_list),
            "max_lat": max(node.lat for node in self.location_data_list),
            "min_lon": min(node.lon for node in self.location_data_list),
            "max_lon": max(node.lon for node in self.location_data_list)
        }
        return self.bounding_box

    def add_nodes_with_coordinates(self, device_list: [Recording_Node]) -> [Recording_Node]:
        """
        Process location data by parsing and converting coordinates.

        Args:
            csv_node_locations (str): Path to CSV file with node locations.
            zone_number (int): UTM zone number (default: 33)
            zone_letter (str): UTM zone letter (default: 'N')

        Returns:
            List[Recording_Node]: Processed locations in specified UTM coordinates.
        """
        self.location_data_list : [Recording_Node] = device_list
        # Update the bounding box
        self.bounding_box = self.__calculate_bounding_box()
        return self.location_data_list

    def add_classifications_for_each_node(self, pkl_file, convert_devIds_to_integer=True) -> None:
        self.data = {}
        self.data = pd.read_pickle(open(os.path.join(pkl_file), "rb"))
        if convert_devIds_to_integer:
            self.data = convert_str_keys_to_int(self.data)

    def set_weight_to_timestamp(self, epoch_ts) -> None:
        """
        Assign each device’s weight based on the confidence at a given timestamp.
        If no entry exists for a device at that timestamp, skip that device.
        """
        self.CLASSIFICATION_TIME_INTERVAL = epoch_ts

        for dev in self.location_data_list:
            df = self.data.get(dev.id)
            if df is None:
                raise IndexError("No data for this device: {dev.id}")

            try:
                # Attempt to grab the first matching confidence value
                dev.weight = df.loc[df['start'] == epoch_ts, 'confidence'].values[0]
            except IndexError:
                # No row matched the timestamp—skip this device
                continue

    def init_graph(self, directedGraph=True) -> None:
        self.G = nx.DiGraph() if directedGraph else nx.Graph()
        for dev in self.location_data_list:
            self.G.add_node(dev.id, node=dev, pos=[dev.lat, dev.lon])
            self.G.nodes[dev.id]['weight'] = dev.weight
            self.G.nodes[dev.id]['label'] = dev.weight


    def delauny(self, e_delta: float = 0.05)-> None:
        """
        Add directed edges to self.G based on Delaunay triangulation of node positions.

        Performs a Delaunay triangulation over the (lat, lon) coordinates in
        self.location_data_list, then for each triangle edge:
          1. Adds a directed edge from the higher-weight node to the lower-weight node.
          2. If the absolute weight difference ≤ e_delta, also adds the reverse edge.
        Edge weights are set to the rounded absolute difference of node weights.

        Parameters
        ----------
        e_delta : float, optional
            Maximum allowed weight difference for inserting a reverse edge (default: 0.05).

        Returns
        -------
        None
        """

        def add_edge_with_condition(node_u, node_v, index_to_node_id):
            u_id = index_to_node_id[node_u]
            v_id = index_to_node_id[node_v]

            # Retrieve weights and compute absolute difference (rounded)
            w_u = self.G.nodes[u_id]['weight']
            w_v = self.G.nodes[v_id]['weight']
            diff = round(abs(w_u - w_v), 3)

            # Add edge from higher‐weight to lower‐weight node
            if w_u > w_v and not self.G.has_edge(u_id, v_id):
                self.G.add_edge(u_id, v_id, weight=diff)

            # If weights are similar, add the reverse edge
            elif diff <= e_delta and not self.G.has_edge(v_id, u_id):
                self.G.add_edge(v_id, u_id, weight=diff)

        # 1. Extract (lat, lon) coordinates into an array for triangulation
        points = np.array([
            (node.lat, node.lon)
            for node in self.location_data_list
        ])

        # 2. Build a lookup for Delaunay: index -> original node ID
        index_to_node_id = {
            idx: node.id
            for idx, node in enumerate(self.location_data_list)
        }

        # 3. Compute the Delaunay triangulation over our sensor locations
        triangulation = Delaunay(points)

        # 4. For each triangle (simplex), examine its three edges
        for simplex in triangulation.simplices:
            # Edges of a triangle: (0->1), (1->2), (2->0)
            for a, b in [(0, 1), (1, 2), (2, 0)]:
                src_idx = simplex[a]
                dst_idx = simplex[b]

                # Attempt to add an edge src -> dst if it meets our weight-based conditions
                add_edge_with_condition(src_idx, dst_idx, index_to_node_id)
                # Also try the reverse direction (dst -> src) if within tolerance
                add_edge_with_condition(dst_idx, src_idx, index_to_node_id)

    def remove_long_edges(self, d_max : float = 100.0) -> nx.Graph:
        for edge in list(self.G.edges):
            distance = eucl_dist(self.G.nodes[edge[0]]['pos'], self.G.nodes[edge[1]]['pos'])
            if distance > d_max:
                self.G.remove_edge(edge[0], edge[1])
        return self.G

    def tase( self, threshold_R: float = 0.5, threshold_B: float = 0.1, TS_delta: float = 0.1,
              threshold_T: float = np.inf) -> Dict[Any, Dict[str, Any]]:
        """
        Apply the Territorial Acoustic Species Estimation (TASE) algorithm.
        Returns dict[root] = {'TS': subgraph, 'TS_weight': total_weight}.
        """
        extractor = self.TASE(outer=self, threshold_R=threshold_R, threshold_B=threshold_B, TS_delta=TS_delta,
                              threshold_T=threshold_T)
        return extractor.run()

    class TASE:
        """
        Inner helper class encapsulating all of the TASE algorithm’s state and methods.
        """

        def __init__(self, outer: "CustomizedGraph", threshold_R: float, threshold_B: float, TS_delta: float,
                     threshold_T: float):
            self.outer = outer
            self.threshold_R = threshold_R
            self.threshold_B = threshold_B
            self.TS_delta = TS_delta
            self.threshold_T = threshold_T

        def deepcopy(self, G: nx.Graph) -> nx.Graph:
            """
            Return a new graph with the same nodes, edges, and node attributes. Library networkx only do shallow copies.

            Each node’s attribute dict is a fresh dict, but attribute values
            themselves are not deep-copied (mutable values remain shared).

            Parameters
            ----------
            G : nx.Graph
                Graph to copy.

            Returns
            -------
            nx.Graph
                Copy of G with duplicated node attribute dicts.
            """
            newG = G.copy()
            for n in G.nodes():
                newG.nodes[n].update(G.nodes[n])
            return newG

        def is_valid_TS(self, G: nx.DiGraph, path):
            """Return True if the path’s border weight, center-to-border distance, and weight monotonicity meet thresholds."""
            # 1. Border‐node weight check
            border_node = path[-1]
            if G.nodes[border_node]['weight'] < self.threshold_B:
                return False

            # 2. Spatial constraint: center‐to‐border distance
            center_node = path[0]
            center_pos = G.nodes[center_node]['pos']
            border_pos = G.nodes[border_node]['pos']
            distance = eucl_dist(center_pos, border_pos)
            if distance > self.threshold_T:
                return False

            # 3. Weight monotonicity within tolerance
            for current_node, next_node in zip(path, path[1:]):
                if G.nodes[next_node]['weight'] - G.nodes[current_node]['weight'] > self.TS_delta:
                    return False

            return True

        def extract_subgraph(self, G: nx.Graph, root: Any) -> nx.DiGraph:
            """
            BFS from `root`, adding neighbors whose path satisfies is_valid_TS.
            """
            # Keep track of which nodes have already been added to the subgraph / visited
            visited = {root}

            # Initialize the resulting territorial subgraph as a directed graph
            TS = nx.DiGraph()
            # Add the root node with its weight and position attributes
            TS.add_node(
                root,
                weight=G.nodes[root]['weight'],
                pos=G.nodes[root]['pos']
            )

            # Store, for each node, the sequence of predecessors from the root up to but not including that node
            pred: Dict[Any, List[Any]] = {root: []}

            # Standard BFS queue, seeded with the root node
            queue = deque([root])

            # Process until there are no more nodes to explore
            while queue:
                cur = queue.popleft()  # dequeue next node to explore

                # Examine each neighbor of the current node
                for nbr in G.neighbors(cur):
                    # Skip if we've already visited this neighbor
                    if nbr in visited:
                        continue

                    # Build the full path of nodes from root -> ... -> cur -> nbr
                    path = pred[cur] + [cur, nbr]

                    # If the path doesn't satisfy the TS validity criteria, skip this neighbor
                    if not self.is_valid_TS(G, path):
                        continue

                    # Mark neighbor as visited, so we don't enqueue it again
                    visited.add(nbr)

                    # Add the neighbor node to the subgraph (copying weight and position)
                    TS.add_node(
                        nbr,
                        weight=G.nodes[nbr]['weight'],
                        pos=G.nodes[nbr]['pos']
                    )
                    # Add the edge from current node → neighbor
                    TS.add_edge(cur, nbr)

                    # Record the path taken to reach this neighbor (for future validity checks)
                    pred[nbr] = pred[cur] + [cur]

                    # Enqueue the neighbor so its neighbors will be explored in turn
                    queue.append(nbr)

            # Return the constructed subgraph containing all valid TS nodes/edges
            return TS

        def run(self) -> Dict[Any, Dict[str, Any]]:
            """
            Execute the TASE extraction workflow.

            Steps:
              1. Deep‐copy the original graph to avoid mutating it.
              2. Sort all nodes by descending weight and filter those above threshold_R as root candidates.
              3. For each root:
                 a. Extract its territorial subgraph (TS) via BFS and validity checks.
                 b. Remove any nodes in that TS from the root candidate list to prevent overcounting.
                 c. Compute the total weight of the TS. This is part of the algorithm for legacy reasons.
              4. Return a mapping from each root node to its TS graph and total weight.

            Returns:
                Dict[Any, Dict[str, Any]]:
                    Keys are root node IDs; values are dicts with:
                      - 'TS': the extracted nx.DiGraph of that root’s territory
                      - 'TS_weight': sum of the territory’s node weights
            """
            # 1) Work on a copy so original graph remains untouched
            G_copy = self.deepcopy(self.outer.G)

            # 2) Identify root candidates: nodes with weight ≥ threshold_R, sorted by weight
            sorted_nodes = sorted(
                G_copy.nodes(data=True),
                key=lambda x: x[1]['weight'],
                reverse=True
            )
            R = [node_id for node_id, attr in sorted_nodes
                 if attr['weight'] >= self.threshold_R]

            results: Dict[Any, Dict[str, Any]] = {}

            # 3) Iterate through root candidates until none remain
            while R:
                root = R.pop(0)

                # 3a) Extract the territorial subgraph for this root
                ts_graph = self.extract_subgraph(G_copy, root)

                # Skip if extraction yielded no valid nodes
                if ts_graph.number_of_nodes() < 1:
                    continue

                # 3b) Remove TS nodes from future root consideration
                for n in ts_graph.nodes():
                    if n in R:
                        R.remove(n)

                # 3c) Sum up weights of all nodes in this TS
                total_w = sum(
                    G_copy.nodes[n]['weight']
                    for n in ts_graph.nodes()
                )

                # Store the subgraph and its total weight under this root
                results[root] = {
                    'TS': ts_graph,
                    'TS_weight': total_w
                }

            # 4) Return mapping of roots → TS graphs and weights
            return results
