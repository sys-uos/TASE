import math
from dataclasses import dataclass, field


@dataclass
class Parameters:
    e_delta: float = 0.2
    e_threshold_meter: float = 700
    threshold_R: float= 0.5
    threshold_B: float= 0.1
    TS_delta: float= 0.7
    threshold_T: float = math.sqrt(70000 / math.pi) * 1.5  # assumption 7ha, based on radios sqrt(7ha / pi). Radius is unrealistic, hence multiply with 1.5, which is about 298.54

    def to_string(self, delimiter: str = "-") -> str:
        """
        Converts the parameters of the class into a formatted string.

        :param delimiter: Delimiter used to separate key-value pairs (default: ", ")
        :return: A string representation of all parameters
        """
        params = {attr: getattr(self, attr) for attr in dir(self) if
                  not attr.startswith("__") and not callable(getattr(self, attr))}

        def format_value(value):
            if isinstance(value, float):
                return f"{value:.2f}".rstrip("0").rstrip(".")  # Formats to 2 decimal places, removes trailing zeroes
            return str(value)

        return delimiter.join(f"{key}={format_value(value)}" for key, value in params.items())

