import os
import pickle

from TASE.src.core.localization import calculate_weighted_centroid_from_nxgraph
from TASE.src.core.tase import CustomizedGraph
from TASE.plotting.deployment.parameters import get_TASE_ParameterSet
from TASE.plotting.deployment.utils import deployment_duration
from TASE.parsing import parse_audiomoth_locations
from TASE.plotting.deployment.species import Phoenicurs_phoenicurus, evaluation_specs
from TASE.src.models import Recording_Node
from TASE.src.utils import convert_wgs84_to_utm


def apply_tase_for_all_20230603(spec=Phoenicurs_phoenicurus(), font_size=12):
    # --- Define deployment duration --- #
    deployment_start, deployment_end = deployment_duration()

    # --- Parse Node Locations --- #
    csv_node_locations = "./TASE/data/20230603/processed/locations/Audiomoth_DeploymentIDs2AudiomothIDs.csv"
    node_locations: [Recording_Node] = parse_audiomoth_locations(csv_node_locations)
    location_data_list = convert_wgs84_to_utm(node_locations, zone_number=32, zone_letter='N')

    # --- Define output directory --- #
    fpath = "./TASE/data/20230603/processed/tase"
    os.makedirs(fpath, exist_ok=True)

    # --- Apply TASE with different set of parameters --- #
    for spec in evaluation_specs():
        dir_classification = f"./TASE/data/20230603/processed/classifications/species_specific/{spec.lat_name.replace(' ', '_')}"
        pkl_dir = f"./TASE/data/20230603/processed/classifications/pkl/{spec.lat_name.replace(' ', '_')}"
        filename = "-".join(os.path.normpath(dir_classification).split(os.sep)[-1:]) + ".pkl"

        for params in get_TASE_ParameterSet(spec):  # --- Define Parameters of TASE --- #
            output_dir = f"./TASE/data/20230603/processed/tase/{os.path.normpath(dir_classification).split(os.sep)[-1]}/"
            ofilename = params.to_string(delimiter="-") + ".pkl"
            if os.path.exists(os.path.join(output_dir, ofilename)):
                continue

            territorial_subgraphs_all = {}  # key: epoch-timestamp, value: territorial subgraphs
            for ts in range(int(deployment_start) + 0, int(deployment_end)-2, 1):  # int(deployment_end)-2 because the last sample starts 1 seconds prior recording end
                graph = CustomizedGraph()
                graph.add_nodes_with_coordinates(device_list=location_data_list)
                graph.add_classifications_for_each_node(pkl_file=os.path.join(pkl_dir, filename))
                graph.set_weight_to_timestamp(ts)
                graph.init_graph(directedGraph=True)
                graph.delauny(e_delta=params.e_delta)
                graph.remove_long_edges(d_max=params.d_max)
                territorial_subgraphs = graph.tase(threshold_R=params.threshold_R,
                                                   threshold_B=params.threshold_B,
                                                   threshold_T=params.threshold_T,
                                                   TS_delta=params.TS_delta)
                # --- Estimate the birds location --- #
                for root in territorial_subgraphs:
                    territorial_subgraphs[root]['location'] = calculate_weighted_centroid_from_nxgraph(territorial_subgraphs[root]['TS'])
                territorial_subgraphs_all[ts] = territorial_subgraphs

                # ---Create some output to be sure that some computation is going one ;-) --- #
                if (ts-deployment_start) % 900 == 0:
                    print(f"Classified ({spec.lat_name}): {ts-deployment_start} of {deployment_end -deployment_start}")

            output_dir = f"./TASE/data/20230603/processed/tase/{os.path.normpath(dir_classification).split(os.sep)[-1]}/"
            ofilename = params.to_string(delimiter="-") + ".pkl"
            os.makedirs(os.path.join(output_dir), exist_ok=True)

            # --- Save the results to disc ---#
            with open(os.path.join(output_dir, ofilename), "wb") as f:
                pickle.dump(territorial_subgraphs_all, f)

            print(f"Saved  for specie {spec.lat_name} data to {os.path.join(output_dir, ofilename)}")