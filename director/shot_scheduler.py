from __future__ import annotations

from typing import Dict, List


class ShotScheduler:
    def schedule(self, camera_plans: List[Dict[str, str]]) -> List[Dict[str, str]]:
        timeline = []
        cursor = 0.0
        for shot in camera_plans:
            item = dict(shot)
            duration = float(item.get("duration", 4.0))
            item["start"] = round(cursor, 3)
            item["end"] = round(cursor + duration, 3)
            cursor += duration
            timeline.append(item)
        return timeline
