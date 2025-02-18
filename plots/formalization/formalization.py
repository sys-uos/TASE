import os
import pickle

import networkx as nx
import numpy as np

from TASE.algo.params import Parameters
from TASE.algo.tase import BirdEstimatorDirected
from TASE.deployment.parameters import get_TASE_ParameterSet
from TASE.deployment.utils import deployment_duration
from TASE.parsing import parse_audiomoth_locations
from TASE.plots.range_analysis.range_analysis import plot_range_analysis
from TASE.deployment.species import Phoenicurs_phoenicurus, Sylvia_atricapilla
from TASE.src.models import Recording_Node
from TASE.src.utils import convert_wgs84_to_utm
from TASE.viewer import WMSMapViewer

def plot_formalization(spec=Phoenicurs_phoenicurus(), font_size=12):
    # --- Define deployment duration --- #
    deployment_start, deployment_end = deployment_duration()

    # --- Parse Node Locations --- #
    csv_node_locations = "./data/20230603/processed/locations/Audiomoth_DeploymentIDs2AudiomothIDs.csv"
    node_locations: [Recording_Node] = parse_audiomoth_locations(csv_node_locations)
    location_data_list = convert_wgs84_to_utm(node_locations, zone_number=32, zone_letter='N')

    # --- Define output directory --- #
    fpath = "./plots/final/formalization/"
    os.makedirs(fpath, exist_ok=True)

    points = [[node.lat, node.lon] for node in location_data_list]
    points = np.array(points)

    # --- Create the WMSMapViewer instance --- #
    viewer = WMSMapViewer()
    viewer.display_with_voronoi(points, figpath=os.path.join(fpath, "voronoi.pdf"),
                                font_size=font_size)

    # --- Define Parameters of TASE --- #
    params = Parameters(
        threshold_R=0.5,
        threshold_B=0.1,
        TS_delta=0.2,
        e_threshold_meter=100,
        threshold_T=spec.max_root_2_leaf_distance(),
        e_delta=0.2,
    )

    # --- Build path to the classification --- #
    dir_classification = f"./data/20230603/processed/classifications/species_specific/1.5_0/{spec.lat_name.replace(' ', '_')}/"
    output_dir = f"./data/20230603/processed/classifications/pkl/{os.path.normpath(dir_classification).split(os.sep)[-1]}"
    filename = "-".join(os.path.normpath(dir_classification).split(os.sep)[-2:-1]) + ".pkl"
    pkl_file = os.path.join(output_dir, filename)

    # --- Build graph and perform tase --- #
    graph = BirdEstimatorDirected()
    graph.add_nodes_with_coordinates(device_list=location_data_list)
    graph.add_classifications_for_each_node(pkl_file=pkl_file)
    graph.set_weight_to_timestamp(deployment_start+50)
    graph.init_graph(directedGraph=True)
    # for illustration's sake, slightly modify the existing node weights
    node_weights = {22: 0.83, 2: 0.41, 3: 0.0, 25: 0.89, 24: 0.75,
                    14: 0.48, 9: 0.57, 10: 0.6, 29: 0.21}  # Node 1 has weight 5.0, node 3 has weight 10.0
    nx.set_node_attributes(graph.G, node_weights, name='weight')
    graph.delauny(e_delta=params.e_delta)
    graph.remove_long_edges(threshold_meter=params.e_threshold_meter)

    # --- Create the WMSMapViewer instance --- #
    viewer = WMSMapViewer()
    viewer.display_with_graph(graph.G, {}, figpath=os.path.join(fpath, "delauny.pdf"),
                              no_legend=True, font_size=font_size)

    territorial_subgraphs = graph.tase(threshold_R=params.threshold_R,
                                       threshold_B=params.threshold_B,
                                       threshold_T=params.threshold_T,
                                       TS_delta=params.TS_delta)

    # --- Plot the graph after modified BFS --- #
    viewer = WMSMapViewer()
    viewer.display_with_graph(graph.G, territorial_subgraphs,
                              figpath=os.path.join(fpath, "delauny_with_subgrpahs.pdf"), font_size=font_size)

    # --- Plot Pointcloud and Heatmap --- #
    spec = Sylvia_atricapilla()
    params = get_TASE_ParameterSet(spec)[0]
    params_string = params.to_string(delimiter='-')
    fn_spec = spec.lat_name.replace(" ", "_")
    path = f"./data/20230603/processed/tase/{fn_spec}/{params_string}.pkl"

    # --- Define output directory --- #
    odir = f"./plots/final/formalization/"
    os.makedirs(odir, exist_ok=True)

    with open(path, "rb") as f:
        territorial_subgraphs_all = pickle.load(f)

        # Extracting centroids and timestamps
        timestamps, centroids = [], []
        for timestamp, node_data in territorial_subgraphs_all.items():
            for node_id, attributes in node_data.items():
                if 'location' in attributes:
                    timestamps.append(timestamp)
                    centroids.append(attributes['location'])

        # Create the WMSMapViewer instance
        viewer = WMSMapViewer()
        viewer.add_circleset_from_utm(centerset=spec.ground_truth)
        viewer.convert_pointcloudUTM_2_pointcloudPIXEL(centroids, timestamps)
        viewer.add_node_locations(node_locations, zone_number=32, zone_letter='N')
        viewer.display_with_pointcloud(deployment_start, deployment_end,
                                       figpath=os.path.join(odir, f"point_cloud.pdf"))
        viewer.display_with_heatmap(font_size=22, figpath=os.path.join(odir, f"heat_map.pdf"), bw_method=spec.bw,
                                    heatmap_vmax=spec.heatmap_vmax)
