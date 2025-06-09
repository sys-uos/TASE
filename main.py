import os.path
import pytz
import datetime

from TASE.plotting.deployment.main import evaluation_of_deployment_20230603
from TASE.plotting.deployment.parsing import check_and_fill_missing_entries, \
    save_classification_data
from TASE.plotting.deployment.species import Sylvia_atricapilla
from TASE.src.core.localization import calculate_weighted_centroid_from_nxgraph
from TASE.src.core.params import Parameters
from TASE.src.core.tase import CustomizedGraph
from TASE.src.utils.tase_utils import extract_locations
from TASE.parsing.classification import parse_classifications_as_dir, \
    add_date_to_classification_dataframe
from TASE.parsing.location import parse_audiomoth_locations
from TASE.src.models import Recording_Node
from TASE.src.utils import convert_wgs84_to_utm
from TASE.viewer.WmsVisualizer import WMSMapViewer


def main_minimal_usage_example():
    #  --- This minimal working example uses a subset of data from a real-world deployment from the publication. --- #

    # 0. Define path to classification data directory and node locations CSV file
    dir_classification = "./TASE/example/data/processed/classifications/species_specific/Sylvia_atricapilla/"
    csv_node_locations = "./TASE/example/data/processed/locations/locations.csv"

    # 1. Define species accordint to /src/core/species.py
    spec = Sylvia_atricapilla()

    # 2. Parse Node Locations and convert to coordinate system
    node_locations_wgs84: [Recording_Node] = parse_audiomoth_locations(csv_node_locations)
    node_locations_utm = convert_wgs84_to_utm(node_locations_wgs84, zone_number=32, zone_letter='N')

    # 3. Define deployment duration (only a subset due to computation time)
    berlin_tz = pytz.timezone("Europe/Berlin")
    dt_start = datetime.datetime(2023, 6, 3, 9, 55, 0)  # 3rd June 2023, 09:00 in Berlin (CEST)
    dt1_aware = berlin_tz.localize(dt_start)  # Make it timezone-aware
    dt_end = datetime.datetime(2023, 6, 3, 10, 0, 0)  # 3rd June 2023, 10:00 in Berlin (CEST)
    dt2_aware = berlin_tz.localize(dt_end)
    deployment_start, deployment_end = dt1_aware.timestamp(), dt2_aware.timestamp()

    # 4. Parse Classifier Results and time in classification file
    # Note: memory consumption can get quite high for many nodes, so for each node a pkl is saved to disc
    out_pkl_file = os.path.join("./TASE/example/data/processed/classifications/pkl", spec.lat_name + ".pkl")
    if not os.path.exists(out_pkl_file):
        dict_devid_df = parse_classifications_as_dir(dir_path=dir_classification)  # key: devid, value: dataframe
        dict_devid_df = add_date_to_classification_dataframe(dict_devid_df, deployment_start)
        dict_devid_df = check_and_fill_missing_entries(dict_devid_df)  # there shouldn't be any, but make sure no gaps exist
        save_classification_data(dict_devid_df, out_pkl_file)

    # 5. Set Parameters for TASE
    params = Parameters(
        threshold_R=0.8,
        threshold_B=0.1,
        TS_delta=0.2,
        threshold_T=spec.max_root_2_leaf_distance(),
        d_max=100,
    )
    # 6.1 Create customized graph
    graph = CustomizedGraph()  # by default, nothing happens here
    graph.add_nodes_with_coordinates(device_list=node_locations_utm)  # add nodes, including their coordinates
    graph.add_classifications_for_each_node(pkl_file=out_pkl_file)  # each node contains a list of their classification results

    # 6.2 Apply TASE for each window starting at deployment_start to deployment_end (refers to seconds)
    territorial_subgraphs_all = {}  # key: epoch-timestamp, value: territorial subgraphs
    for ts in range(int(deployment_start), int(deployment_end)-7, 1):  #  5 s gap between recordings + 3 s BirdNET gap = 8s time of last classification window
        start_dt = datetime.datetime.fromtimestamp(ts, tz=berlin_tz)
        end_dt = datetime.datetime.fromtimestamp(ts+3, tz=berlin_tz)  # the value 3 refers to 3 seconds, which is BirdNET's window
        print(f"Apply TASE on period from {start_dt} to {end_dt}")
        graph.init_graph(directedGraph=True)  # init the graph with the locations
        graph.set_weight_to_timestamp(ts)  # set the weights of the node to a specific window
        graph.delauny(e_delta=0.2)  # perform delauny algorithm, add edge only if condition is met
        graph.remove_long_edges(d_max=params.d_max)  # remove edges that exceed 100m
        # 6.3 Apply TASE-Algorithm to extract territorial subgraphs
        territorial_subgraphs = graph.tase(threshold_R=params.threshold_R,
                                           threshold_B=params.threshold_B,
                                           threshold_T=params.threshold_T,
                                           TS_delta=params.TS_delta)

        # 6.3 Derive representation for each territorial subgraph --- #
        for root in territorial_subgraphs:
            territorial_subgraphs[root]['location'] = calculate_weighted_centroid_from_nxgraph(territorial_subgraphs[root]['TS'])
        territorial_subgraphs_all[ts] = territorial_subgraphs

    print("Visualize results using WMS-Service...")

    # 7. Visualize the results
    dict_ts_centroids = extract_locations(territorial_subgraphs_all)  # for visualization, resulting in key: ts, value: centroid


    # A custom class for visualization with a background map is used that integrated wms-services
    viewer = WMSMapViewer(wms_service={"url": "http://www.wms.nrw.de/geobasis/wms_nw_dtk",
                                       "version": "1.3.0",
                                       "layername": "nw_dtk_col"})
    viewer.add_circleset_from_utm(centerset=spec.ground_truth)
    viewer.add_and_convert_pointcloudUTM_2_pointcloudPIXEL(dict_ts_centroids, zone_number=32, zone_letter='N')
    viewer.add_circleset_from_utm(centerset=spec.ground_truth)
    viewer.add_node_locations(node_locations_utm, zone_number=32, zone_letter='N')
    viewer.display_with_pointcloud(point_size=20,font_size=27,alpha=0.5, figpath="./TASE/example/example_pointcloud.png")
    viewer.display_with_heatmap_and_groundtruth(font_size=25, bw_method=spec.bw, heatmap_vmax=spec.heatmap_vmax,
                                                figpath="./TASE/example/example_heatmap.png", alpha=0.5)
    print("Heatmap saved at: ./TASE/example/example_heatmap.pdf")
    print("Minimal Example ended successfully!")


if __name__ == "__main__":
    # To run a minimal working example
    main_minimal_usage_example()

    # To reproduce the results of the paper, check the following function:
    evaluation_of_deployment_20230603()

