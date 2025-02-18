import os
import pickle

import pandas as pd


def fill_missing_timeintervals_with_zero(device_list, df_classifications, start_end_of_deployment=[1684800004, 1694566787],
                                         output_dir=None, override=False):  # interval_2023 = [1684800004, 1694566787]
    if output_dir is None:
        output_dir = "./data/classifications/pkl"
    os.makedirs(output_dir, exist_ok=True)

    for dev in device_list:
        print("Device: ", dev.id)
        output_path = os.path.join(output_dir, f"{dev.id}.pkl")
        if os.path.exists(output_path) and not override:
            print(f"File exists, skipping: {output_path}")
            continue

        df = df_classifications[df_classifications['id'] == dev.id][['id', 'date', 'beginn', 'start', 'end', 'confidence', ]]
        df['datetime'] = pd.to_datetime(df_classifications['date'], format='%Y%m%d')

        # convert value in column start (of recording) to seconds
        def time_to_seconds(time):
            time_str = str(time)
            if len(time_str) == 1:
                minutes = int(time_str[:1])
                total_seconds = (minutes * 60)
            elif len(time_str) == 2:
                minutes = int(time_str[:2])
                total_seconds = (minutes * 60)
            elif len(time_str) == 3:
                hours = int(time_str[:1])
                minutes = int(time_str[2:])
                total_seconds = (hours * 3600) + (minutes * 60)
            elif len(time_str) == 4:
                hours = int(time_str[:2])
                minutes = int(time_str[2:])
                total_seconds = (hours * 3600) + (minutes * 60)
            return total_seconds

        # Add the 'start' column as hours to the datetime (assuming 'start' is in hours)
        df['datetime'] = df['datetime'] + pd.to_timedelta(df['beginn'].apply(time_to_seconds),
                                                        unit='s') + pd.to_timedelta(df['start'], unit='s')

        # Convert the resulting datetime to epoch time (seconds since 1970-01-01)
        df['start_epoch'] = df['datetime'].astype(int) // 10 ** 9  # converting to Unix timestamp in seconds

        # Display the updated DataFrame
        df = df[['id', 'start_epoch', 'confidence']]

        # Create a complete sequence of epoch times within this range
        full_range = pd.DataFrame({'start_epoch': range(start_end_of_deployment[0], start_end_of_deployment[1] + 1)})

        # Merge with the original DataFrame to find missing epochs
        df_full = full_range.merge(df, on='start_epoch', how='left')

        # Fill missing confidence values with 0.0
        df_full['confidence'] = df_full['confidence'].fillna(0.0)
        df_full['id'] = df_full['id'].fillna(dev.id)

        # Fill the missing 'id' column with the original id (assuming it's the same for all rows)
        df_full['id'] = df_full['id'].fillna(method='ffill')  # Forward-fill method if the id is continuous

        output_path = os.path.join(output_dir, f"{dev.id}.pkl")
        print(f"Saving data for device {dev.id} to: {output_path}")
        pickle.dump(df_full, open(output_path, "wb"))


def fill_missing_entries(df: pd.DataFrame, time_difference_in_samples: int, sample_rate=48000) -> pd.DataFrame:
    """
    Fill missing entries in the DataFrame where the difference between sequential 'start' times
    is greater than the expected time difference in samples. Missing rows are inserted with
    'confidence' set to 0.0 and the same species_common.

    :param df: Pandas DataFrame with 'start' column as timestamps
    :param time_difference_in_samples: The expected time difference in samples
    :param sample_rate: The sample rate in Hz (default is 48000)
    :return: A new DataFrame with missing entries filled
    """
    # Convert sample difference to seconds
    expected_time_diff = time_difference_in_samples / sample_rate

    # Create a new list to store the corrected rows
    new_rows = []

    for i in range(len(df) - 1):
        new_rows.append(df.iloc[i])  # Add the current row

        # Calculate the difference between the current and next start time
        time_diff = df.iloc[i + 1]["start"] - df.iloc[i]["start"]

        # If the difference is greater than expected, insert missing rows
        while time_diff > expected_time_diff:
            missing_start = new_rows[-1]["start"] + expected_time_diff
            missing_end = missing_start + (df.iloc[i]["end"] - df.iloc[i]["start"])  # Keep duration the same

            missing_row = df.iloc[i].copy()  # Copy an existing row for consistency
            missing_row["start"] = missing_start
            missing_row["end"] = missing_end
            missing_row["confidence"] = 0.0  # Set confidence to 0.0

            new_rows.append(missing_row)  # Append as a DataFrame row

            # Update the remaining time difference
            time_diff -= expected_time_diff

    # Add the last row of the original DataFrame
    new_rows.append(df.iloc[-1].copy())

    # Convert list of Series back into a DataFrame
    filled_df = pd.DataFrame(new_rows)

    # Reset index
    filled_df.reset_index(drop=True, inplace=True)

    return filled_df


def check_time_difference(df: pd.DataFrame, time_difference_in_samples: int, sample_rate=48000) -> bool:
    """
    Check if the difference between sequential rows in the 'start' column
    is exactly time_difference_in_samples.

    :param df: Pandas DataFrame with 'start' column as timestamps
    :param time_difference_in_samples: The expected time difference in samples
    :param sample_rate: The sample rate in Hz (default is 48000)
    :return: True if all sequential differences match, False otherwise
    """
    # Convert samples to seconds
    expected_time_diff = time_difference_in_samples / sample_rate

    # Compute the actual time differences
    actual_diffs = df["start"].diff().dropna()

    # Check if all differences match the expected difference
    return (actual_diffs == expected_time_diff).all()