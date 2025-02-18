import os

import pandas as pd
from datetime import datetime

import pytz


def convert_timestamp_to_datetime(timestamp: float, timezone: str = "Europe/Berlin") -> datetime:
    """
    Convert a timestamp (e.g., 1.685758e+09) to a datetime object in the specified timezone.

    :param timestamp: Unix timestamp (seconds since epoch, can be in scientific notation)
    :param timezone: Timezone string (default is "UTC")

    :return: Datetime object in the specified timezone
    """
    # Convert timestamp to UTC datetime
    utc_dt = datetime.utcfromtimestamp(timestamp)
    # Convert to the specified timezone
    target_tz = pytz.timezone(timezone)
    localized_dt = pytz.utc.localize(utc_dt).astimezone(target_tz)
    return localized_dt


def add_date_to_classification_dataframe(dict_devid_df, epochtime_offset, sample_rate=48000) -> pd.DataFrame:
    for key, df_value in dict_devid_df.items():
        df_value["start"] = df_value["start"] / sample_rate + epochtime_offset
        df_value["end"] = df_value["end"] / sample_rate + epochtime_offset
    return dict_devid_df

def parse_classifications_as_dir(dir_path,
                                column_mapping=None):
    all_data = {}

    for node_dir in os.listdir(dir_path):
        node_path = os.path.join(dir_path, node_dir)
        if os.path.isdir(node_path):
            node_data = []
            current_offset = 0

            i = 0
            for file in sorted(os.listdir(node_path)):
                # print(file)
                if file.endswith('.BirdNET.results.txt'):
                    file_path = os.path.join(node_path, file)
                    df = pd.read_csv(file_path, header=None,
                                     names=['start', 'end', 'species_latin', 'species_common', 'confidence'],
                                     sep=',')

                    df['start'] += current_offset
                    df['end'] += current_offset
                    current_offset = df['end'].max() + (48000 * 5)

                    # print(df)

                    node_data.append(df)
                    # if i == 6:
                    #     break
                    # i += 1

            if node_data:
                all_data[node_dir] = pd.concat(node_data, ignore_index=True)
                # break

        # break
    # print(all_data)
    # print(all_data['01'].loc[3591])
    # print(all_data['01'].loc[3592])
    return all_data


def parse_classifications_as_file(csv_classifications,
                                column_mapping=None):
    """
    Read CSV classifications file and rename columns according to mapping.

    Args:
        csv_classifications (str): Path to CSV file containing classifications
        column_mapping (dict, optional): Dictionary mapping original column names to new names.
            Defaults to mapping sound.files->file, geraet->id, datum->date, x->longitude, y->latitude

    Returns:
        pandas.DataFrame: DataFrame with renamed columns according to mapping
    """
    if column_mapping is None:
        column_mapping = {
            'sound.files': 'file',
            'geraet': 'id',
            'datum': 'date',
            'x': 'longitude',
            'y': 'latitude'
        }
    df = pd.read_csv(csv_classifications, sep=',')
    df = df.rename(columns=column_mapping)
    return df