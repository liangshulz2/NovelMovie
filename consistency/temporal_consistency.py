from __future__ import annotations

from typing import Dict, List


class TemporalConsistency:
    def smooth(self, timeline: List[Dict[str, str]]) -> List[Dict[str, str]]:
        prev_end = 0.0
        for shot in timeline:
            shot["start"] = max(float(shot.get("start", 0.0)), prev_end)
            duration = float(shot.get("duration", 4.0))
            shot["end"] = shot["start"] + duration
            prev_end = shot["end"]
        return timeline
