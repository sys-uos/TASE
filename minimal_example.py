import os.path
import pytz
import datetime

from TASE.plotting.deployment.parsing import check_and_fill_missing_entries, \
    save_classification_data
from TASE.plotting.deployment.species import Sylvia_atricapilla
from TASE.src.core.localization import calculate_weighted_centroid_from_nxgraph
from TASE.src.core.params import Parameters
from TASE.src.core.tase import CustomizedGraph
from TASE.src.utils.tase_utils import extract_locations
from parsing.classification import parse_classifications_as_dir, \
    add_date_to_classification_dataframe
from parsing.location import parse_audiomoth_locations
from src.models import Recording_Node
from src.utils import convert_wgs84_to_utm
from TASE.viewer.WmsVisualizer import WMSMapViewer


def main_minimal_usage_example():
    #  --- This minimal working example uses a subset of data from a real-world deployment from the publication. --- #

    # 0. Define path to classification data directory and node locations CSV file
    dir_classification = "./example/data/processed/classifications/species_specific/Sylvia_atricapilla/"
    csv_node_locations = "./example/data/processed/locations/locations.csv"

    # 1. Load species data such as mean territory size and ground-truth data, according to /src/core/species.py
    spec = Sylvia_atricapilla()

    # 2. Load Node Locations:
    node_locations: [Recording_Node] = parse_audiomoth_locations(csv_node_locations)
    location_data_list = convert_wgs84_to_utm(node_locations, zone_number=32, zone_letter='N')

    # 3. Define deployment duration (only a subset due to computation time)
    berlin_tz = pytz.timezone("Europe/Berlin")
    dt_start = datetime.datetime(2023, 6, 3, 7, 9, 50) # 3rd June 2023, 04:00 in Berlin (CEST)
    dt1_aware = berlin_tz.localize(dt_start)  # Make it timezone-aware
    dt_end = datetime.datetime(2023, 6, 3, 7, 10, 0)  # 3rd June 2023, 6:10 in Berlin (CEST)
    dt2_aware = berlin_tz.localize(dt_end)
    deployment_start, deployment_end  = dt1_aware.timestamp(), dt2_aware.timestamp()

    # 4. Parse Classifier Results and time in classification file
    # Note: memory consumption can get quite high for many nodes, so for each node a pkl is saved to disc
    out_pkl_file = os.path.join("./example/data/processed/classifications/pkl", spec.lat_name + ".pkl")
    if not os.path.exists(out_pkl_file):
        dict_devid_df = parse_classifications_as_dir(dir_path=dir_classification)  # key: devid, value: dataframe
        dict_devid_df = add_date_to_classification_dataframe(dict_devid_df, deployment_start)
        dict_devid_df = check_and_fill_missing_entries(dict_devid_df)
        save_classification_data(dict_devid_df, out_pkl_file)

    # 5. Define Parameters of TASE
    params = Parameters(
        threshold_R=0.8,
        threshold_B=0.1,
        TS_delta=0.2,
        threshold_T=300,
    )
    # 6. Create Graph
    graph = CustomizedGraph()
    graph.add_nodes_with_coordinates(device_list=location_data_list)
    graph.add_classifications_for_each_node(pkl_file=out_pkl_file)
    territorial_subgraphs_all = {}  # key: epoch-timestamp, value: territorial subgraphs
    for ts in range(int(deployment_start) + 0, int(deployment_end) - 3, 1):
        print(f"Apply TASE on Epoch-Time {ts} to {ts+3}")
        graph.init_graph(directedGraph=True)
        graph.set_weight_to_timestamp(ts)
        graph.delauny(e_delta=0.2)
        graph.remove_long_edges(threshold_meter=700.0)
        # 7. Apply TASE-Algorithm to extract subgraphs
        territorial_subgraphs = graph.tase(threshold_R=params.threshold_R,
                                           threshold_B=params.threshold_B,
                                           threshold_T=params.threshold_T,
                                           TS_delta=params.TS_delta)

        # --- Estimate the birds location and append them to location --- #
        for root in territorial_subgraphs:
            territorial_subgraphs[root]['location'] = calculate_weighted_centroid_from_nxgraph(territorial_subgraphs[root]['TS'])
        territorial_subgraphs_all[ts] = territorial_subgraphs

    dict_ts_centroids = extract_locations(territorial_subgraphs_all)

    # --- Visualize the results --- #
    viewer = WMSMapViewer(wms_service={"url": "http://www.wms.nrw.de/geobasis/wms_nw_dtk", "version": "1.3.0", "layername": "nw_dtk_col"})
    viewer.add_and_convert_pointcloudUTM_2_pointcloudPIXEL(dict_ts_centroids, zone_number=32, zone_letter='N')
    viewer.add_node_locations(location_data_list)
    viewer.display_with_pointcloud(point_size=50, font_size=16, figpath="./example/example_pointcloud.png")
    viewer.display_with_heatmap(font_size=16, bw_method=0.2, heatmap_vmax=0.0001, figpath="./example/example_heatmap.png")
    exit(0)




if __name__ == "__main__":
    main_minimal_usage_example()
