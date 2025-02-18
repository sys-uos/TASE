import networkx as nx
import numpy as np
from pyproj import Transformer, Proj, transform

def convert_wgs84_to_utm(location_data_list, zone_number=None, zone_letter='N'):
    # Initialize transformer for UTM zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{32600 + zone_number + (0 if zone_letter >= 'N' else 100)}",
                                       always_xy=True)

    for dev in location_data_list:
        try:
            utm_easting, utm_northing = transformer.transform(dev.lon, dev.lat)
            dev.lat, dev.lon = [round(utm_easting, 2), round(utm_northing, 2)]
        except Exception:
            pass

    print(f"converted locations to UTM{zone_number}{zone_letter}")

    return location_data_list


def eucl_dist(xyz1 : [], xyz2: []):
    point1 = np.array(xyz1)
    point2 = np.array(xyz2)
    return np.linalg.norm(point1 - point2)


def calc_geometric_center_in_Graph(G: nx.Graph, cluster_nodes, weighted=False):
    # Initialize variables to store the weighted sum of x and y coordinates
    sum_weighted_x = 0.0
    sum_weighted_y = 0.0
    sum_weights = 0.0

    # Iterate through the nodes with positions
    for node in cluster_nodes:
        node_position = G.nodes[node]['pos']
        if weighted:
            # Calculate weighted geometric mean
            node_weight = G.nodes[node]['weight']
            # print(node_position[0])
            sum_weighted_x += np.log(node_position[0]) * node_weight
            sum_weighted_y += np.log(node_position[1]) * node_weight
            sum_weights += node_weight
        else:
            # Calculate unweighted geometric mean
            sum_weighted_x += np.log(node_position[0])
            sum_weighted_y += np.log(node_position[1])

    if weighted and sum_weights == 0.0:
        return np.inf, np.inf

    # Calculate the geometric center coordinates
    center_x = np.exp(sum_weighted_x / sum_weights) if weighted else np.exp(sum_weighted_x / len(cluster_nodes))
    center_y = np.exp(sum_weighted_y / sum_weights) if weighted else np.exp(sum_weighted_y / len(cluster_nodes))

    return center_x, center_y
