
def convert_str_keys_to_int(d: dict) -> dict:
    """
    Convert all dictionary keys that are numeric strings to integers.

    :param d: Dictionary with string keys
    :return: Dictionary with numeric string keys converted to integers
    """
    return {int(k) if k.isdigit() else k: v for k, v in d.items()}
