import pytz
import datetime

from TASE.src.models import Recording_Node
from TASE.src.utils import convert_wgs84_to_utm
from TASE.parsing import parse_audiomoth_locations


def deployment_duration():
    """
        Calculates the start and end timestamps of a deployment period in Berlin timezone.

        This function defines a deployment period starting on **June 3, 2023, at 04:00 AM CEST**
        and ending on **June 3, 2023, at 10:00 AM CEST**. It localizes the datetime objects to
        Berlin's timezone (Europe/Berlin) and returns the corresponding Unix timestamps.

        Returns:
        --------
        tuple[float, float]
            A tuple containing:
            - deployment_start (float): Unix timestamp of the start time.
            - deployment_end (float): Unix timestamp of the end time.
    """
    berlin_tz = pytz.timezone("Europe/Berlin")
    dt_start = datetime.datetime(2023, 6, 3, 4, 0, 0) # 3rd June 2023, 04:00 in Berlin (CEST)
    dt1_aware = berlin_tz.localize(dt_start)  # Make it timezone-aware
    deployment_start = dt1_aware.timestamp()
    dt_end = datetime.datetime(2023, 6, 3, 10, 0, 0)  # 3rd June 2023, 10:00 in Berlin (CEST)
    dt2_aware = berlin_tz.localize(dt_end)
    deployment_end = dt2_aware.timestamp()
    return deployment_start, deployment_end


def deployment_node_locations():
    """
    Loads and processes the locations of recording nodes from a CSV file,
    converts their coordinates from WGS84 to UTM format, and returns the processed data.

    Returns:
    --------
    list
        A list of location data in UTM format, obtained by converting
        the parsed `Recording_Node` objects.
    """
    csv_node_locations = "./data/20230603/processed/locations/Audiomoth_DeploymentIDs2AudiomothIDs.csv"
    node_locations: [Recording_Node] = parse_audiomoth_locations(csv_node_locations)
    location_data_list = convert_wgs84_to_utm(node_locations, zone_number=32, zone_letter='N')
    return location_data_list
