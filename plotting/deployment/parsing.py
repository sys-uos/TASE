from TASE.plotting.deployment.utils import deployment_duration
from TASE.parsing.classification import add_date_to_classification_dataframe, parse_classifications_as_dir
from TASE.plotting.deployment.species import evaluation_specs
from TASE.src.utils.classifications_utils import check_time_difference, fill_missing_entries


import os
import pickle
import pandas as pd

def parse_classification_data(spec, deployment_start, deployment_end):
    """
    Parses classification data for a given species, adds timestamps, and ensures missing start/end data is filled.

    Parameters:
    -----------
    spec : object
        The species object containing classification directory information.
    deployment_start : int
        The deployment start timestamp.
    deployment_end : int
        The deployment end timestamp.

    Returns:
    --------
    dict
        A dictionary where keys are device IDs and values are pandas DataFrames with parsed classification data.
    """
    dir_classification = f"./data/20230603/processed/classifications/species_specific/{spec.lat_name.replace(' ', '_')}/"
    dict_devid_df = parse_classifications_as_dir(dir_path=dir_classification)
    dict_devid_df = add_date_to_classification_dataframe(dict_devid_df, deployment_start)

    # Add missing lines for each device at the start
    for devid, df in dict_devid_df.items():
        missing_entries = int(df['start'].iloc[0] - deployment_start)
        if missing_entries < 0:
            raise Exception("Something went wrong during parsing (Start time mismatch).")

        new_rows = []
        base_timestamp = deployment_start
        base_species_latin = df['species_latin'].iloc[0]
        base_species_common = df['species_common'].iloc[0]

        for _ in range(missing_entries):
            new_rows.append({
                'start': base_timestamp,
                'end': base_timestamp + 3,
                'species_latin': base_species_latin,
                'species_common': base_species_common,
                'confidence': 0.0
            })
            base_timestamp += 1

        dict_devid_df[devid] = pd.concat([pd.DataFrame(new_rows), df], ignore_index=True)

    # Add missing lines for each device at the end
    for devid, df in dict_devid_df.items():
        missing_entries = int(deployment_end - df['end'].iloc[-1])
        if missing_entries < 0:
            raise Exception("Something went wrong during parsing (End time mismatch).")

        new_rows = []
        base_timestamp = df['start'].iloc[-1]
        base_species_latin = df['species_latin'].iloc[-1]
        base_species_common = df['species_common'].iloc[-1]

        for _ in range(missing_entries):
            new_timestamp = base_timestamp + 1  # 1-second increments
            new_rows.append({
                'start': new_timestamp,
                'end': new_timestamp + 3,
                'species_latin': base_species_latin,
                'species_common': base_species_common,
                'confidence': 0.0
            })
            base_timestamp = new_timestamp

        dict_devid_df[devid] = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    return dict_devid_df


def check_and_fill_missing_entries(dict_devid_df, time_difference_in_samples=48000):
    """
    Checks classification results for missing entries and fills them where necessary.

    Parameters:
    -----------
    dict_devid_df : dict
        A dictionary where keys are device IDs and values are pandas DataFrames with classification data.
    time_difference_in_samples : int, optional (default=48000)
        Time difference threshold in samples to check for missing entries.

    Returns:
    --------
    dict
        The updated classification dictionary with missing entries filled if needed.
    """
    print("Checking classification results for missing entries...")
    print(f"DevIds:", end=" ")
    for devid, df in dict_devid_df.items():
        print(f"{devid}", end=', ')
        if not check_time_difference(df, time_difference_in_samples):
            dict_devid_df[devid] = fill_missing_entries(df, time_difference_in_samples)
    print()
    return dict_devid_df


def save_classification_data(dict_devid_df, filepath):
    """
    Saves processed classification data to a pickle file.

    Parameters:
    -----------
    dict_devid_df : dict
        A dictionary where keys are device IDs and values are pandas DataFrames with classification data.
    dir_classification : str
        Directory where the classification files are stored.

    Returns:
    --------
    None
    """
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(dict_devid_df, f)

    print(f"Data written to {filepath}")


def parse_data_from_20230603():
    """
    Main function to parse, check, and save classification data for all species in the dataset.
    """
    deployment_start, deployment_end = deployment_duration()

    for spec in evaluation_specs():
        dict_devid_df = parse_classification_data(spec, deployment_start, deployment_end)
        dict_devid_df = check_and_fill_missing_entries(dict_devid_df)
        output_dir = f"./data/20230603/processed/classifications/pkl/{spec.lat_name.replace(' ', '_')}"
        filename = spec.lat_name.replace(' ', '_') + ".pkl"
        fpath = os.path.join(output_dir, filename)
        save_classification_data(dict_devid_df, fpath)
