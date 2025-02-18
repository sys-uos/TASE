import os
import pickle

import pandas as pd

from TASE.deployment.utils import deployment_duration
from TASE.parsing import parse_audiomoth_locations
from TASE.parsing.classification import add_date_to_classification_dataframe, parse_classifications_as_dir
from TASE.deployment.species import Phoenicurs_phoenicurus, evaluation_specs
from TASE.src.models import Recording_Node
from TASE.src.utils import convert_wgs84_to_utm
from TASE.src.utils.classifications_utils import check_time_difference, fill_missing_entries


def parse_data_from_20230603():
    # --- Define deployment duration --- #
    deployment_start, deployment_end = deployment_duration()

    # --- Parse Node Locations --- #
    csv_node_locations = "./data/20230603/processed/locations/Audiomoth_DeploymentIDs2AudiomothIDs.csv"
    node_locations: [Recording_Node] = parse_audiomoth_locations(csv_node_locations)
    location_data_list = convert_wgs84_to_utm(node_locations, zone_number=32, zone_letter='N')

    # --- Parse Classifier Results --- #
    # Note: memory consumption can get quite high for many nodes, so for each node a pkl is saved to disc
    for spec in evaluation_specs()[-2:-1]:
        dir_classification = f"./data/20230603/processed/classifications/species_specific/1.5_0/{spec.lat_name.replace(' ', '_')}/"

        dict_devid_df = parse_classifications_as_dir(dir_path=dir_classification)
        dict_devid_df = add_date_to_classification_dataframe(dict_devid_df, deployment_start)

        # Add missing lines for each devid at the *start*, given the deployment_start
        for devid in dict_devid_df:
            missing_entries = int(dict_devid_df[devid]['start'].iloc[0]- deployment_start)
            if missing_entries < 0:
                raise Exception("Something went wrong during parsing")
            new_rows = []
            base_timestamp = deployment_start  # Get the timestamp of the start
            base_species_latin = dict_devid_df[devid]['species_latin'].iloc[0]  # Get the last timestamp
            base_species_common = dict_devid_df[devid]['species_common'].iloc[0]  # Get the last timestamp
            for i in range(missing_entries):  # Add 3 new rows as an example
                new_row = {
                    'start': base_timestamp,
                    'end': base_timestamp+3,
                    'species_latin': base_species_latin,
                    'species_common': base_species_common,
                    'confidence': 0.0
                }
                new_rows.append(new_row)
                base_timestamp = base_timestamp + 1
            dict_devid_df[devid] = pd.concat([pd.DataFrame(new_rows), dict_devid_df[devid]], ignore_index=True)

        # Add missing lines for each devid at the end, given the deployment_end
        for devid in dict_devid_df:
            missing_entries = int(deployment_end - dict_devid_df[devid]['end'].iloc[-1])
            if missing_entries < 0:
                raise Exception("Something went wrong during parsing")
            new_rows = []
            base_timestamp = dict_devid_df[devid]['start'].iloc[-1]  # Get the last timestamp
            base_species_latin = dict_devid_df[devid]['species_latin'].iloc[-1]  # Get the last timestamp
            base_species_common = dict_devid_df[devid]['species_common'].iloc[-1]  # Get the last timestamp
            for i in range(missing_entries):  # Add 3 new rows as an example
                new_timestamp = base_timestamp + 1  # 1 refers to seconds
                new_row = {
                    'start': new_timestamp,
                    'end': new_timestamp + 3,
                    'species_latin': base_species_latin,
                    'species_common': base_species_common,
                    'confidence': 0.0
                }
                new_rows.append(new_row)
                base_timestamp = new_timestamp
            dict_devid_df[devid] = pd.concat([dict_devid_df[devid], pd.DataFrame(new_rows)], ignore_index=True)

        # --- Assure that the results do not contain any missing entries  --- #
        print("Check classification results for missing entries...")
        print(f"DevIds:", end=" ")
        for id, df in dict_devid_df.items():
            print(f"{id}", end=', ')
            if not check_time_difference(df, time_difference_in_samples=48000):
                dict_devid_df[id] = fill_missing_entries(dict_devid_df[id], time_difference_in_samples=48000)

        output_dir = f"./data/20230603/processed/classifications/pkl/{os.path.normpath(dir_classification).split(os.sep)[-1]}"
        filename = "-".join(os.path.normpath(dir_classification).split(os.sep)[-2:-1]) + ".pkl"
        os.makedirs(os.path.join(output_dir), exist_ok=True)
        with open(os.path.join(output_dir, filename), "wb") as f:
            pickle.dump(dict_devid_df, f)
        print(f"Written data to {os.path.join(output_dir, filename)}")