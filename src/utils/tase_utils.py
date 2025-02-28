def extract_locations(territorial_subgraphs: {}):
    """
    Extracts locations from a nested dictionary structure where timestamps are keys.

    Parameters:
    -----------
    territorial_subgraphs : dict
        A dictionary where keys are timestamps and values are nested dictionaries
        containing location data, e.g.
        {1685764801: {14: {'TS': <networkx.classes.digraph.DiGraph object at 0x7f18790a5cd0>, 'TS_weight': 3.4096, 'location': (435113.412975129, 5762641.484516952)}}}

    Returns:
    --------
    dict
        A dictionary with timestamps as keys and a list of locations as values.
    """
    extracted_data = {}
    for timestamp, nested_dict in territorial_subgraphs.items():
        locations = [
            value['location']
            for key, value in nested_dict.items()
            if 'location' in value
        ]
        extracted_data[timestamp] = locations
    return extracted_data