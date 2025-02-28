import os.path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from TASE.src.models import Recording_Node
from TASE.src.utils import eucl_dist, convert_str_keys_to_int


class BirdEstimatorDirected:

    def __init__(self):
        pass

    def __calculate_bounding_box(self):
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

    def add_classifications_for_each_node(self, pkl_file, convert_devIds_to_integer=True):
        self.data = {}
        self.data = pd.read_pickle(open(os.path.join(pkl_file), "rb"))
        if convert_devIds_to_integer:
            self.data = convert_str_keys_to_int(self.data)


    def set_weight_to_timestamp(self, epoch_ts):
        self.CLASSIFICATION_TIME_INTERVAL = epoch_ts
        for dev in self.location_data_list:
            df = self.data[dev.id]
            dev.weight = df.loc[df['start'] == epoch_ts, 'confidence'].values[0]

    def init_graph(self, directedGraph=True):
        self.G = nx.DiGraph() if directedGraph else nx.Graph()
        for dev in self.location_data_list:
            self.G.add_node(dev.id, node=dev, pos=[dev.lat, dev.lon])
            self.G.nodes[dev.id]['weight'] = dev.weight
            self.G.nodes[dev.id]['label'] = dev.weight


    def delauny(self, e_delta: float = 0.05):

        def add_edge_with_condition(node_u, node_v, lookup_points_devId):
            node_u = lookup_points_devId[node_u]
            node_v = lookup_points_devId[node_v]
            weight_u = self.G.nodes[node_u]['weight']
            weight_v = self.G.nodes[node_v]['weight']
            weight_diff = abs(weight_u - weight_v)

            if weight_u > weight_v:
                if not self.G.has_edge(node_u, node_v):
                    self.G.add_edge(node_u, node_v, weight=round(abs(weight_diff), 3))

            if weight_diff <= e_delta:
                if not self.G.has_edge(node_v, node_u):
                    self.G.add_edge(node_v, node_u, weight=round(abs(weight_diff), 3))

        # Convert node coordinates
        points = np.array([(node.lat, node.lon) for node in self.location_data_list])
        lookup_points_devId = {}
        for i, node in enumerate(self.location_data_list):
            lookup_points_devId[i] = node.id

        # Perform Delaunay triangulation
        triangulation = Delaunay(points)

        # Add edges from the Delaunay triangulation
        for simplex in triangulation.simplices:
            for i in range(3):
                node1 = simplex[i]
                node2 = simplex[(i + 1) % 3]
                add_edge_with_condition(node1, node2, lookup_points_devId)
                add_edge_with_condition(node2, node1, lookup_points_devId)  # Consider both directions if the condition is met

    def remove_long_edges(self, threshold_meter : float = 500.0):
        for edge in list(self.G.edges):
            distance = eucl_dist(self.G.nodes[edge[0]]['pos'], self.G.nodes[edge[1]]['pos'])
            if distance > threshold_meter:
                self.G.remove_edge(edge[0], edge[1])
        return self.G



    def tase(self, threshold_R: float = 0.5, threshold_B: float = 0.1, TS_delta: float = 0.1, threshold_T=np.inf):

        def copy_with_G_with_weights(G: nx.Graph):
            newG = G.copy()
            for nodeid in newG.nodes:
                for key in G.nodes[nodeid].keys():
                    newG.nodes[nodeid][key] = G.nodes[nodeid][key]
            return newG

        def check_criteria(G: nx.DiGraph, path, leaf_threshold: float, weight_threshold: float,
                           breeding_birds_distance):
            # the latest added node is always the last one

            # 1. Criteria: Check if score is almost zero
            for node in path:
                if G.nodes[node]['weight'] < leaf_threshold:
                    return False
            # 2. Criteria: Check if the weights are declining
            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]
                weight_diff = G.nodes[current_node]['weight'] - G.nodes[next_node]['weight']
                # print(current_node, next_node, weight_diff)
                if G.nodes[current_node]['weight'] - G.nodes[next_node]['weight'] < (-1) * weight_threshold:
                    return False
            # 3. Criteria: distance between the latest node to the first node (which is the center)
            distance = eucl_dist(G.nodes[path[0]]['pos'], G.nodes[path[-1]]['pos'])
            # print(path[0], path[-1], distance)
            if distance > breeding_birds_distance:
                return False
            return True

        def bfs_with_all_predecessors(G: nx.Graph, root, leaf_threshold: float, weight_threshold: float,
                                      breeding_birds_distance):
            from collections import deque

            # mark the visited nodes
            visited = set()

            # remeber the predecessors in the bfs for checking the criteria
            predecessors = {root: []}  # Start node has no predecessors

            # the cluster, starting with node start
            subgraph = nx.DiGraph()  # Initialize the subgraph as a NetworkX graph
            subgraph.add_node(root, weight=G.nodes[root]['weight'], pos=G.nodes[root]['pos'])

            # Initialize the queue with the start node
            queue = deque([root])

            while queue:
                # Dequeue the current node
                current_node = queue.popleft()

                # mark the current node as visited
                visited.add(current_node)

                # neighbors are sorted in ascending order
                neighbors_id = G.neighbors(current_node)
                neighbors = {}
                for n in neighbors_id:
                    neighbors[n] = {'weight': G.nodes[n]['weight']}

                for neighbor in neighbors:
                    if neighbor not in visited and \
                            check_criteria(G, predecessors[current_node] + [current_node, neighbor],
                                                leaf_threshold,
                                                weight_threshold, breeding_birds_distance):
                        visited.add(neighbor)

                        # Add the edge to the subgraph if the neighbor has not been visited before
                        subgraph.add_node(neighbor, weight=G.nodes[neighbor]['weight'],
                                          pos=G.nodes[neighbor]['pos'])
                        subgraph.add_edge(current_node, neighbor)
                        predecessors[neighbor] = predecessors[current_node] + [
                            current_node]  # Update predecessors for the neighbor
                        queue.append(neighbor)  # Enqueue the neighbor for further exploration

            return subgraph

        self.G_ = copy_with_G_with_weights(self.G)

        # -- Prepare the root node for each territorial subgraph --- #t
        # sort nodes by their weight
        R = [elem[0] for elem in sorted(self.G_.nodes(data=True), key=lambda x: x[1]['weight'], reverse=True)]
        R = [elem for elem in R if self.G_.nodes[elem]['weight'] >= threshold_R]
        # print("R: ", R)

        # datastructure for the territorial subgraphs
        TS = {}  # key: staring_node, value: cluster

        while len(R) != 0:
            # select starting node for modified bfs
            root = R.pop(0)

            # perform bfs and get cluster with root as starting node
            ts: nx.DiGraph = bfs_with_all_predecessors(self.G_, root, threshold_B, TS_delta,
                                                             threshold_T)
            if ts is None: # when is this the case?
                continue

            # nodes in R that are already part of a TS cannot be the start of another TS
            for id in ts.nodes:
                try:
                    R.remove(id)
                except ValueError:
                    continue

            # calculate some information of the clusters that might be used later
            ts_weight = 0.0
            for i, id in enumerate(ts.nodes):
                ts_weight += self.G_.nodes[id]['weight']

            TS[root] = {'TS': ts, 'TS_weight': ts_weight}

        return TS


