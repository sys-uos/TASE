import pytz
import datetime

from TASE.src.models import Recording_Node
from TASE.src.utils import convert_wgs84_to_utm
from TASE.parsing import parse_audiomoth_locations


def deployment_duration():
    # --- Define deployment duration --- #
    berlin_tz = pytz.timezone("Europe/Berlin")
    dt_start = datetime.datetime(2023, 6, 3, 4, 0, 0) # 3rd June 2023, 04:00 in Berlin (CEST)
    dt1_aware = berlin_tz.localize(dt_start)  # Make it timezone-aware
    deployment_start = dt1_aware.timestamp()
    dt_end = datetime.datetime(2023, 6, 3, 10, 0, 0)  # 3rd June 2023, 10:00 in Berlin (CEST)
    dt2_aware = berlin_tz.localize(dt_end)
    deployment_end = dt2_aware.timestamp()
    return deployment_start, deployment_end

def deployment_node_locations():
    # --- Parse Node Locations --- #
    csv_node_locations = "./data/20230603/processed/locations/Audiomoth_DeploymentIDs2AudiomothIDs.csv"
    node_locations: [Recording_Node] = parse_audiomoth_locations(csv_node_locations)
    location_data_list = convert_wgs84_to_utm(node_locations, zone_number=32, zone_letter='N')
    return location_data_list
