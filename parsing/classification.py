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
    """
        Adjusts the start and end times in multiple classification DataFrames by converting
        sample-based timestamps to epoch-based timestamps.

        Parameters:
        ----------
        dict_devid_df : dict[str, pd.DataFrame]
            A dictionary where keys are device IDs (strings) and values are Pandas DataFrames.
            Each DataFrame must contain 'start' and 'end' columns with values in sample indices.

        epochtime_offset : float
            The epoch time (Unix timestamp) that serves as the reference point for conversion.

        sample_rate : int, optional (default=48000)
            The sample rate in Hz, used to convert sample indices into seconds.

        Returns:
        -------
        dict_devid_df : dict[str, pd.DataFrame]
            The same dictionary with updated DataFrames where 'start' and 'end' times
            are converted to absolute epoch time.
    """
    for key, df_value in dict_devid_df.items():
        df_value["start"] = df_value["start"] / sample_rate + epochtime_offset
        df_value["end"] = df_value["end"] / sample_rate + epochtime_offset
    return dict_devid_df


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


def parse_classifications_as_dir(dir_path):
    """
    Parse BirdNET classification results from a directory tree.

    Parameters
    ----------
    dir_path : str
        Path to the root directory containing subdirectories for each node.
        Each node directory should contain one or more `.BirdNET.results.txt`
        files with no header, comma-separated columns:
            0: start (int)
            1: end (int)
            2: species_latin (str)
            3: species_common (str)
            4: confidence (float)

    Returns
    -------
    dict[str, pandas.DataFrame]
        A dictionary mapping node directory names to DataFrames of concatenated
        classification results. Each DataFrame has columns:
          - 'start': start time in samples, adjusted by offsets
          - 'end':   end time in samples, adjusted by offsets
          - 'species_latin': scientific name of detected species
          - 'species_common': common name of detected species
          - 'confidence': detection confidence score

   Notes
    -----
    - A 5-second gap between consecutive recordings is assumed, at a sample rate
      of 48 000 Hz (i.e. 5 * 48000 samples).
    - After reading each file, `current_offset` is increased by the maximum
      'end' sample plus the fixed gap to ensure time continuity across files.

    """
    all_data = {}

    for node_dir in sorted(os.listdir(dir_path)):
        node_path = os.path.join(dir_path, node_dir)
        if os.path.isdir(node_path):
            node_data = []
            current_offset = 0

            for file in sorted(os.listdir(node_path)):
                if file.endswith('.BirdNET.results.txt'):
                    file_path = os.path.join(node_path, file)
                    df = pd.read_csv(file_path, header=None,
                                     names=['start', 'end', 'species_latin', 'species_common', 'confidence'],
                                     sep=',')

                    df['start'] += current_offset
                    df['end'] += current_offset
                    current_offset = df['end'].max() + (48000 * 5)  # latter term because of the recording gap between the audios for saving the audio
                    node_data.append(df)
            if node_data:
                all_data[node_dir] = pd.concat(node_data, ignore_index=True)
    return all_data