from __future__ import annotations

from typing import Dict, List


class CameraPlanner:
    def plan(self, directed_shots: List[Dict[str, str]]) -> List[Dict[str, str]]:
        plans = []
        for shot in directed_shots:
            plan = dict(shot)
            plan["camera_path"] = "start: static -> move: dolly-in -> end: close-up"
            plan["lens"] = "35mm"
            plans.append(plan)
        return plans
