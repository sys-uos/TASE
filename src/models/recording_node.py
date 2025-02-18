from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Recording_Node:
    id: str
    lat: float
    lon: float
    ele: float = 0.0
    weight: float = -1.0
