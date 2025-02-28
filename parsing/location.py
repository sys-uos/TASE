from typing import List
import pandas as pd

from TASE.src.models import Recording_Node


def parse_recording_locations(csv_file: str="./data/preprocessed/node_locations.csv") -> List[Recording_Node]:
    """
    Load recording node locations from a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing location data.

    Returns:
        List[Recording_Node]: A list of Recording_Node objects with
        parsed location information from the CSV.

    CSV File Structure:
    - Columns: 'id', 'latitude', 'longitude'
    - 'id': Unique string identifier
    - 'latitude': Decimal degrees (float, range -90 to 90)
    - 'longitude': Decimal degrees (float, range -180 to 180)

    Example of csv_file:
        id,latitude,longitude
        recording_001,40.7128,-74.0060
        recording_002,34.0522,-118.2437
    """
    df = pd.read_csv(csv_file)
    location_data_list = [Recording_Node(row['id'], row['latitude'], row['longitude'])
                          for _, row in df.iterrows()]
    return location_data_list

def parse_audiomoth_locations(csv_file: str="./data/preprocessed/node_locations.csv") -> List[Recording_Node]:
    """
    Load recording node locations from a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing location data.

    Returns:
        List[Recording_Node]: A list of Recording_Node objects with
        parsed location information from the CSV.

    CSV File Structure:
    - Columns: 'id', 'latitude', 'longitude'
    - 'id': Unique string identifier
    - 'latitude': Decimal degrees (float, range -90 to 90)
    - 'longitude': Decimal degrees (float, range -180 to 180)

    Example of csv_file:
        DeploymentIDs,AudiomothIDs,,GPS_ID,"m_lat [WGS84 Ellipsiod, WGS84 Koordsystem]","m_lon  [WGS84 Ellipsiod, WGS84 Koordsystem]","m_elevation  [WGS84 Ellipsiod, WGS84 Koordsystem]",above_ground [cm]
        1,2474750763FA466E,,A01,52.009779,8.05353967,66.31633867,126
    """
    df = pd.read_csv(csv_file)
    df = df.rename(columns={'m_lat [WGS84 Ellipsiod, WGS84 Koordsystem]': 'm_lat',
                            'm_lon  [WGS84 Ellipsiod, WGS84 Koordsystem]': 'm_lon'})
    location_data_list: [Recording_Node] = [Recording_Node(row['DeploymentIDs'], row['m_lat'], row['m_lon'])
                          for _, row in df.iterrows()]
    return location_data_list
