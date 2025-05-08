import os
import pickle
from datetime import datetime

import pytz
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from TASE.src.core.params import Parameters
from TASE.src.core.tase import CustomizedGraph
from TASE.plotting.deployment.parameters import get_TASE_ParameterSet
from TASE.plotting.deployment.species import evaluation_specs, Phoenicurs_phoenicurus, Turdus_philomelos, Troglodytes_troglodytes
from TASE.plotting.deployment.utils import deployment_node_locations, deployment_duration
from TASE.parsing import parse_audiomoth_locations
from TASE.src.models import Recording_Node
from TASE.src.utils import convert_wgs84_to_utm
from TASE.plotting.viewer import WMSMapViewer


def make_plots_of_species_heatmaps():
    # --- Get Node Locations --- #
    node_locations = deployment_node_locations()

    for spec in evaluation_specs()[7:8]:
        fn_spec = spec.lat_name.replace(" ", "_")

        for params in get_TASE_ParameterSet(spec)[3:]:
            params_string = params.to_string(delimiter='-')
            # path = f"./data/20230603/processed/tase_correct_config/{fn_spec}/{params_string}.pkl"
            path = f"./data/20230603/processed/tase/{fn_spec}/{params_string}.pkl"

            # --- Define output directory --- #
            odir = f"./plotting/plots/evaluation/{fn_spec}/"
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
                figpath = os.path.join(odir, f"{params_string}.pdf")
                viewer = WMSMapViewer()
                viewer.add_circleset_from_utm(centerset=spec.ground_truth)
                viewer.convert_pointcloudUTM_2_pointcloudPIXEL(centroids, timestamps)
                viewer.add_node_locations(node_locations, zone_number=32, zone_letter='N')
                # deployment_start, deployment_end = deployment_duration()
                # viewer.display_with_pointcloud(deployment_start, deployment_end,
                #                                figpath=os.path.join(odir, f"{params_string}.pdf"))
                viewer.display_with_heatmap(font_size=30, figpath=figpath, bw_method=spec.bw,
                                            heatmap_vmax=spec.heatmap_vmax)


def make_plots_about_impact_of_interference():

    def plot_methodological_impact_of_interference(spec, figpath=None, fontsize=30):
        dir_classification = f"./data/20230603/processed/classifications/species_specific/{spec.lat_name.replace(' ', '_')}"
        pkl_dir = f"./data/20230603/processed/classifications/pkl/{os.path.normpath(dir_classification).split(os.sep)[-1]}"
        filename = spec.lat_name.replace(' ', '_') + ".pkl"

        # --- Define deployment duration --- #
        deployment_start, deployment_end = deployment_duration()

        # --- Parse Node Locations --- #
        csv_node_locations = "./data/20230603/processed/locations/Audiomoth_DeploymentIDs2AudiomothIDs.csv"
        node_locations: [Recording_Node] = parse_audiomoth_locations(csv_node_locations)
        location_data_list = convert_wgs84_to_utm(node_locations, zone_number=32, zone_letter='N')

        graph = CustomizedGraph()
        graph.add_nodes_with_coordinates(device_list=location_data_list)
        graph.add_classifications_for_each_node(pkl_file=os.path.join(pkl_dir, filename))
        buckets = {}
        chunk_size = 900  # in seconds, 900s are 15 minutes
        for devid in graph.data.keys():
            # Filter data
            df = graph.data[devid]
            filtered_df = df.loc[
                (df['start'] >= deployment_start) & (df['start'] < (deployment_start + 7200))
                ].copy()
            filtered_df['chunk'] = ((filtered_df['start'] - deployment_start) // chunk_size).astype(int)

            # Group confidence values into buckets
            for chunk, confidences in filtered_df.groupby('chunk')['confidence']:
                if chunk not in buckets:
                    buckets[chunk] = []
                buckets[chunk].extend(confidences.tolist())

        time_intervals = []
        confidence_values = []

        for key, values in buckets.items():
            from_ = datetime.fromtimestamp(deployment_start + key * chunk_size, pytz.timezone('Europe/Berlin')).strftime('%H:%M')
            to_ = datetime.fromtimestamp(deployment_start + (key+1) * chunk_size, pytz.timezone('Europe/Berlin')).strftime('%H:%M')
            time_label = f"{from_} to {to_}"
            time_intervals.extend([time_label] * len(values))
            confidence_values.extend(buckets[key])

        # Create a DataFrame for violin plot
        df = pd.DataFrame({"Time Interval": time_intervals, "Confidence Value": confidence_values})

        plt.figure(figsize=(12, 8))
        plt.axhspan(0.5, 1.0, facecolor='red', alpha=0.25, label=r"Interval of TS roots")
        palette = sns.light_palette("purple", n_colors=len(df["Time Interval"].unique()), reverse=True)
        sns.violinplot(x="Time Interval", y="Confidence Value", data=df[df["Confidence Value"] >= 0.1],
                       palette=palette, scale="width", log_scale=True, inner=None, cut=0)
        plt.minorticks_off()
        plt.xticks(rotation=45, ha="center", fontsize=fontsize)
        plt.yscale('log')
        yticks = np.arange(0, 1.2, 0.1)  # Ticks from 0 to 1 in steps of 0.1
        plt.yticks(yticks, fontsize=fontsize)
        # Format y-ticks in 0.x notation
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # Increase the margin between ticks and the axis
        plt.tick_params(axis="y", pad=15)  # Adds padding between y-ticks and the plot
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)

        plt.xlabel("Time Interval", fontsize=fontsize)
        plt.ylabel("Confidence Score", fontsize=fontsize)
        plt.ylim([0.1, 1])
        plt.legend(loc='upper right', fontsize=fontsize, title_fontsize=fontsize-7)
        plt.tight_layout()

        if figpath:
            plt.savefig(figpath)

        plt.show()

    for spec in [Phoenicurs_phoenicurus(), Turdus_philomelos()]:
        # --- Define output directory --- #
        fn_spec = spec.lat_name.replace(" ", "_")
        odir = f"./plotting/plots/evaluation/Analysis/interference/{fn_spec}/"
        os.makedirs(odir, exist_ok=True)
        figpath = os.path.join(odir, f"{fn_spec}.pdf")

        plot_methodological_impact_of_interference(spec, figpath=figpath)
        print(f"Saved to {figpath}")

def make_plots_about_methodological_challenges(spec=Phoenicurs_phoenicurus(), font_size=27):
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
        d_max=100,
        threshold_T=spec.max_root_2_leaf_distance(),
        e_delta=0.2,
    )

    # --- Build path to the classification --- #
    dir_classification = f"./data/20230603/processed/classifications/species_specific/{spec.lat_name.replace(' ', '_')}"
    output_dir = f"./data/20230603/processed/classifications/pkl/{os.path.normpath(dir_classification).split(os.sep)[-1]}"
    filename = spec.lat_name.replace(' ', '_') + ".pkl"
    pkl_file = os.path.join(output_dir, filename)

    for time_interval in [5, 41, 2268, 2562, 2621]:
        # --- Define output directory --- #
        odir = f"./plotting/plots/evaluation/Analysis/methodological_errors/"
        os.makedirs(odir, exist_ok=True)
        fn_spec = spec.lat_name.replace(" ", "_")
        figpath = os.path.join(odir, f"{fn_spec}_{time_interval}.pdf")


        # --- Build graph and perform tase --- #
        graph = CustomizedGraph()
        graph.add_nodes_with_coordinates(device_list=location_data_list)
        graph.add_classifications_for_each_node(pkl_file=pkl_file)
        graph.set_weight_to_timestamp(deployment_start + time_interval)
        graph.init_graph(directedGraph=True)
        graph.delauny(e_delta=params.e_delta)
        graph.remove_long_edges(threshold_meter=params.d_max)
        territorial_subgraphs = graph.tase(threshold_R=params.threshold_R,
                                           threshold_B=params.threshold_B,
                                           threshold_T=params.threshold_T,
                                           TS_delta=params.TS_delta)

        # --- Plot the graph after modified BFS --- #
        viewer = WMSMapViewer()
        viewer.add_circleset_from_utm(centerset=spec.ground_truth)
        viewer.display_with_graph(graph.G, territorial_subgraphs, font_size=font_size, figpath=figpath)

def make_plots_about_impact_of_timeintervals():
    # --- Define deployment duration --- #
    deployment_start, deployment_end = deployment_duration()

    # --- Parse Node Locations --- #
    csv_node_locations = "./data/20230603/processed/locations/Audiomoth_DeploymentIDs2AudiomothIDs.csv"
    node_locations: [Recording_Node] = parse_audiomoth_locations(csv_node_locations)
    location_data_list = convert_wgs84_to_utm(node_locations, zone_number=32, zone_letter='N')

    spec = Troglodytes_troglodytes()
    fn_spec = spec.lat_name.replace(" ", "_")

    # --- Define Parameters of TASE --- #
    params = Parameters(
        threshold_R=0.5,
        threshold_B=0.1,
        TS_delta=0.2,
        d_max=100,
        threshold_T=spec.max_root_2_leaf_distance(),
        e_delta=0.2,
    )
    params_string = params.to_string(delimiter='-')

    path = f"./data/20230603/processed/tase/{fn_spec}/{params_string}.pkl"

    # --- Define output directory --- #
    odir = f"./plotting/plots/evaluation/{fn_spec}_short_timespans/"
    os.makedirs(odir, exist_ok=True)

    with open(path, "rb") as f:
        territorial_subgraphs_all = pickle.load(f)

        # Define parameters
        chunk_size = 1800  # 30 minutes
        overlap = 900  # 15 minutes
        timestamps, centroids = [], []

        # Convert dictionary keys (timestamps) to a sorted list
        sorted_timestamps = sorted(territorial_subgraphs_all.keys())

        # Determine the first timestamp and set intervals
        current_start = deployment_start + 900  # Initialize window start, +900 because birst start singing later

        # Process data in overlapping intervals
        while current_start <= deployment_start + 2.5 * 3600:
            current_end = current_start + chunk_size  # Define window end

            # Collect timestamps and centroids within the interval
            interval_timestamps, interval_centroids = [], []
            for timestamp in sorted_timestamps:
                if current_start <= timestamp < current_end:
                    for node_id, attributes in territorial_subgraphs_all[timestamp].items():
                        if 'location' in attributes:
                            interval_timestamps.append(timestamp)
                            interval_centroids.append(attributes['location'])

            # Store interval results
            if interval_timestamps:  # Avoid empty intervals
                timestamps.append(interval_timestamps)
                centroids.append(interval_centroids)

            # Move to the next interval with overlap
            current_start += chunk_size - overlap  # Shift window by (chunk_size - overlap)

    for i in range(0, len(timestamps)):
        # Create the WMSMapViewer instance
        figpath = os.path.join(odir, f"{int(deployment_start + (chunk_size - overlap) * i)}_to_"
                                     f"{int(deployment_start + (chunk_size - overlap) * (i+1))}.pdf")
        viewer = WMSMapViewer()
        viewer.add_circleset_from_utm(centerset=spec.ground_truth)
        viewer.convert_pointcloudUTM_2_pointcloudPIXEL(centroids[i], timestamps[i])
        viewer.add_node_locations(node_locations, zone_number=32, zone_letter='N')
        viewer.display_with_heatmap(font_size=30, figpath=figpath, bw_method=spec.bw,
                                    heatmap_vmax=spec.heatmap_vmax)



