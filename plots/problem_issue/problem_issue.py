import os

from TASE.algo.params import Parameters
from TASE.algo.tase import BirdEstimatorDirected
from TASE.deployment.utils import deployment_duration
from TASE.parsing import parse_audiomoth_locations
from TASE.deployment.species import Phoenicurs_phoenicurus
from TASE.src.models import Recording_Node
from TASE.src.utils import convert_wgs84_to_utm
from TASE.viewer import WMSMapViewer


def plot_problem_issue(spec=Phoenicurs_phoenicurus(), font_size=25):
    # --- Define deployment duration --- #
    deployment_start, deployment_end = deployment_duration()

    # --- Parse Node Locations --- #
    csv_node_locations = "./data/20230603/processed/locations/Audiomoth_DeploymentIDs2AudiomothIDs.csv"
    node_locations: [Recording_Node] = parse_audiomoth_locations(csv_node_locations)
    location_data_list = convert_wgs84_to_utm(node_locations, zone_number=32, zone_letter='N')

    # --- Define Parameters of TASE --- #
    params = Parameters(
        threshold_R=0.5,
        threshold_B=0.1,
        TS_delta=0.2,
        threshold_T=100,
        e_delta=0.2,
    )

    # --- Build path to the classification --- #
    dir_classification = f"./data/20230603/processed/classifications/species_specific/1.5_0/{spec.lat_name.replace(' ', '_')}/"
    output_dir = f"./data/20230603/processed/classifications/pkl/{os.path.normpath(dir_classification).split(os.sep)[-1]}"
    filename = "-".join(os.path.normpath(dir_classification).split(os.sep)[-2:-1]) + ".pkl"
    pkl_file = os.path.join(output_dir, filename)

    # --- Define output directory --- #
    fpath = "./plots/final/problem_issue/"
    os.makedirs(fpath, exist_ok=True)

    for ts in range(int(deployment_start+900), int(deployment_start+909)):  # [5] # range(tuple[0], tuple[1]):  #range(30, 100, 1):
        print("CLASSIFICATION_TIME_INTERVAL", ts)
        graph = BirdEstimatorDirected()
        graph.add_nodes_with_coordinates(device_list=location_data_list)
        graph.add_classifications_for_each_node(pkl_file=pkl_file)
        graph.set_weight_to_timestamp(ts)
        graph.init_graph(directedGraph=True)
        graph.delauny(e_delta=params.e_delta)
        graph.remove_long_edges(threshold_meter=params.threshold_T)

        # Create the WMSMapViewer instance
        viewer = WMSMapViewer()
        viewer.add_circleset_from_utm(centerset=spec.ground_truth)
        viewer.display_with_nodes_colored_by_weight(graph.G,
                                  figpath=os.path.join(f"{fpath}", f"{str(ts)}.pdf"),
                                  font_size=font_size)
