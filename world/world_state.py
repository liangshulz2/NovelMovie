from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class WorldState:
    era: str = "现代"
    weather: str = "多云"
    mood: str = "紧张"
    props: Dict[str, str] = field(default_factory=dict)
