import os.path
import pickle

import pytz
import datetime

from TASE.deployment.analysis import apply_tase_for_all_20230603
from TASE.deployment.parsing import parse_data_from_20230603
from TASE.deployment.species import Phoenicurs_phoenicurus
from TASE.plots.evaluation.evaluation import plot_evaluation, plot_evaluation_impact_of_time, \
    plot_methodological_errors, plot_evaluation_spec_over_time
from TASE.plots.formalization.formalization import plot_formalization
from TASE.plots.problem_issue.problem_issue import plot_problem_issue
from TASE.plots.range_analysis.range_analysis import plot_range_analysis
from TASE.viewer import WMSMapViewer
from algo.params import Parameters
from algo.tase import BirdEstimatorDirected
from parsing.classification import parse_classifications_as_file
from parsing.location import parse_audiomoth_locations, parse_recording_locations
from src.models import Recording_Node
from src.utils import convert_wgs84_to_utm
from src.utils.classifications_utils import fill_missing_timeintervals_with_zero


#bed.run_TASE_over_range(start_date_str, end_date_str)


def real_deployment():
    spec = Phoenicurs_phoenicurus()

    dir_classification = f"./data/20230603/processed/classifications/species_specific/1.5_0/{spec.lat_name.replace(' ', '_')}/"
    csv_node_locations = "./data/20230603/processed/locations/Audiomoth_DeploymentIDs2AudiomothIDs.csv"

    # --- Define deployment duration --- #
    berlin_tz = pytz.timezone("Europe/Berlin")
    dt_start = datetime.datetime(2023, 6, 3, 4, 0, 0) # 3rd June 2023, 04:00 in Berlin (CEST)
    dt1_aware = berlin_tz.localize(dt_start)  # Make it timezone-aware
    deployment_start = dt1_aware.timestamp()
    dt_end = datetime.datetime(2023, 6, 3, 10, 0, 0)  # 3rd June 2023, 10:00 in Berlin (CEST)
    dt2_aware = berlin_tz.localize(dt_end)
    deployment_end = dt2_aware.timestamp()
    print(deployment_start, deployment_end)

    # --- Parse Node Locations --- #
    node_locations: [Recording_Node] = parse_audiomoth_locations(csv_node_locations)
    # print(node_locations)
    location_data_list = convert_wgs84_to_utm(node_locations, zone_number=32, zone_letter='N')
    # print(location_data_list)

    # # --- Parse Classifier Results --- #
    # # Note: memory consumption can get quite high for many nodes, so for each node a pkl is saved to disc
    # dict_devid_df = parse_classifications_as_dir(dir_path=dir_classification)
    # dict_devid_df = add_date_to_classification_dataframe(dict_devid_df, deployment_start)
    # # print(dict_devid_df)
    # # print(convert_timestamp_to_datetime(dict_devid_df['01'].iloc[-1]['start']))
    # # print(convert_timestamp_to_datetime(dict_devid_df['01'].iloc[-1]['end']))
    # # print(convert_timestamp_to_datetime(deployment_start))
    #
    # # --- Assure that the results do not contain any missing entries  --- #
    # print("Check classification results for missing entries...")
    # print(f"DevIds:", end=" ")
    # for id, df in dict_devid_df.items():
    #     print(f"{id}", end=', ')
    #     if not check_time_difference(df, time_difference_in_samples=48000):
    #         dict_devid_df[id] = fill_missing_entries(dict_devid_df[id], time_difference_in_samples=48000)

    # df = dict_devid_df['01']
    # filtered_row = df[df["start"] == 1685757604]
    # print(filtered_row)
    # # print(graph.data[1])
    # exit(1)

    # output_dir = f"./data/20230603/processed/classifications/pkl/{os.path.normpath(dir_classification).split(os.sep)[-1]}"
    # filename = "-".join(os.path.normpath(dir_classification).split(os.sep)[-2:-1]) + ".pkl"
    # os.makedirs(os.path.join(output_dir), exist_ok=True)
    # with open(os.path.join(output_dir, filename), "wb") as f:
    #     pickle.dump(dict_devid_df, f)


    # --- Define Parameters of TASE --- #
    params = Parameters(
        threshold_R=0.5,
        threshold_B=0.1,
        TS_delta=0.35,
        threshold_T=300,
        e_delta=0.2,
    )

    output_dir = f"./data/20230603/processed/classifications/pkl/{os.path.normpath(dir_classification).split(os.sep)[-1]}"
    filename = "-".join(os.path.normpath(dir_classification).split(os.sep)[-2:-1]) + ".pkl"

    territorial_subgraphs_all = {}
    for ts in range(int(deployment_start), int(deployment_start) + 10,1):
        # 3. Create Graph
        graph = BirdEstimatorDirected()
        graph.add_nodes_with_coordinates(device_list=location_data_list)
        graph.add_classifications_for_each_node(pkl_file=os.path.join(output_dir, filename))
        graph.set_weight_to_timestamp(ts)
        graph.init_graph(directedGraph=True)
        graph.delauny(e_delta=params.e_delta)
        graph.remove_long_edges(threshold_meter=params.threshold_T)
        territorial_subgraphs = graph.tase(threshold_R=params.threshold_R,
                                           threshold_B=params.threshold_B,
                                           threshold_T=params.threshold_T,
                                           TS_delta=params.TS_delta)
        territorial_subgraphs_all[ts] = territorial_subgraphs

        viewer = WMSMapViewer()
        viewer.display_with_graph(graph.G, territorial_subgraphs)

    output_dir = f"./data/20230603/processed/classifications/pkl/" \
                 f"{os.path.normpath(dir_classification).split(os.sep)[-1]}/tase"
    filename = params.to_string(delimiter="-") + ".pkl"
    os.makedirs(os.path.join(output_dir), exist_ok=True)

    with open(os.path.join(output_dir, filename), "wb") as f:
        pickle.dump(territorial_subgraphs_all, f)

    # viewer = WMSMapViewer()
    # viewer.add_node_locations(node_locations, zone_number=32, zone_letter='N')
    # viewer.display()

def plot_tase_point_cloud(spec=Phoenicurs_phoenicurus()):
    dir_classification = f"./data/20230603/processed/classifications/species_specific/1.5_0/{spec.lat_name.replace(' ', '_')}/"
    csv_node_locations = "./data/20230603/processed/locations/Audiomoth_DeploymentIDs2AudiomothIDs.csv"

    # --- Parse Node Locations --- #
    node_locations: [Recording_Node] = parse_audiomoth_locations(csv_node_locations)
    location_data_list = convert_wgs84_to_utm(node_locations, zone_number=32, zone_letter='N')


    # --- Define Parameters of TASE --- #
    params = Parameters(
        threshold_R=0.5,
        threshold_B=0.1,
        TS_delta=0.35,
        threshold_T=300,
        e_delta=0.2,
    )


    output_dir = f"./data/20230603/processed/classifications/pkl/" \
                 f"{os.path.normpath(dir_classification).split(os.sep)[-1]}/tase"
    filename = params.to_string(delimiter="-") + ".pkl"

    with open(os.path.join(output_dir, filename), "rb") as f:
        data = pickle.load(f)

    print(len(data.keys()))

    viewer = WMSMapViewer()
    viewer.add_node_locations(location_data_list, zone_number=32, zone_letter='N')
    viewer.display()


def minimal_working_example():
    csv_classifications = "/home/caspar/Avis/Zwergohreulen/data_2024/preprocessed/subset_output.csv"
    csv_node_locations = "/home/caspar/Avis/Zwergohreulen/data_2024/preprocessed/Koordinaten24.csv"

    # 1. Parse Node Locations:
    node_locations: [Recording_Node] = parse_recording_locations(csv_node_locations)
    location_data_list = convert_wgs84_to_utm(node_locations, zone_number=33, zone_letter='N')

    # 2. Parse Classifier Results
    # Note: memory consumption can get quite high for many nodes, so for each node a pkl is saved to disc
    df = parse_classifications_as_file(csv_classifications=csv_classifications)
    output_dir = "./data/zwergohreule/2024/preprocessed/classifications/pkl"
    fill_missing_timeintervals_with_zero(location_data_list, df,
                                          output_dir=output_dir)

    # 3. Define Parameters of TASE
    params = Parameters(
        threshold_R=0.8,
        threshold_B=0.1,
        TS_delta=0.35,
        threshold_T=300,
    )

    # 3. Create Graph
    graph = BirdEstimatorDirected()
    graph.add_nodes_with_coordinates(device_list=location_data_list)
    graph.add_classifications_for_each_node(dir=output_dir)
    graph.init_graph(directedGraph=True)
    graph.set_weight_to_timestamp(1684800004)
    graph.delauny(e_delta=0.2)
    graph.remove_long_edges(threshold_meter=700.0)
    territorial_subgraphs = graph.tase(threshold_R=params.threshold_R,
                                      threshold_B=params.threshold_B,
                                      threshold_T=params.threshold_T,
                                      TS_delta=params.TS_delta)
    territorial_subgraphs_all = territorial_subgraphs

    print(territorial_subgraphs_all)
    # dict_dev_weight =
    # graph.set_weights()

    # print(df_full)

def evaluation_of_deployment_20230603():
    # --- Parse data from the deploment --- #
    # parse_data_from_20230603()
    # exit(1)

    # --- Apply TASE on the data --- #
    # apply_tase_for_all_20230603()
    # exit(1)

    # --- Make Plots --- #
    # plot_problem_issue()
    # plot_formalization()
    # plot_range_analysis()
    # plot_evaluation()
    plot_evaluation_spec_over_time()
    # plot_evaluation_impact_of_time()
    # plot_methodological_errors()


if __name__ == "__main__":
    # real_deployment()

    # prepare_data_from_20230603_for_tase()
    # exit(1)
    evaluation_of_deployment_20230603()
    exit(1)
    # TODO: restructering code in plots into dirs

    #plots_problem_issue()
    # plots_Formalization()
    # plots_range_analysis()  # TODO contains too much code
    # TODO Monday: transfer the content of plots_evaluation_20230603(), the graphical abstract, and methodological errors


    # pklfile = "./data/20230603/processed/tase/Anthus_trivialis/TS_delta=0.2-e_delta=0.2-e_threshold_meter=100-threshold_B=0.1-threshold_R=0.5-threshold_T=134.65.pkl"
    pklfile = "./data/20230603/processed/tase/Anthus_trivialis/TS_delta=0.2-e_delta=0.2-e_threshold_meter=100-threshold_B=0.1-threshold_R=0.5-threshold_T=134.65.pkl"
    with open(pklfile, "rb") as f:
        territorial_subgraphs_all = pickle.load(f)

    ctr = 0
    territorial_subgraphs_all_mod = {}
    for ts in territorial_subgraphs_all:
        territorial_subgraphs_all_mod[ts] = {}
        if len(territorial_subgraphs_all[ts].keys()) == 0:
            continue
        elif len(territorial_subgraphs_all[ts].keys()) >= 1:
            print(territorial_subgraphs_all[ts])
            for root in territorial_subgraphs_all[ts].keys():
                if len(territorial_subgraphs_all[ts][root]['TS'].nodes) >= 1:
                    # territorial_subgraphs_all_mod[ts] = {}
                    ctr += 1
                # print(territorial_subgraphs_all[ts][root]['TS'].nodes)
                # print()
        # if len(territorial_subgraphs_all[ts].keys()) == 1:
        #     ctr += 1
        #     print(territorial_subgraphs_all[ts].keys(), len(territorial_subgraphs_all[ts].keys()))
    print(ctr)
    exit(1)
