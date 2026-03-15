from __future__ import annotations

from typing import Dict, List


class DirectorAgent:
    def __init__(self, style_controller) -> None:
        self.style_controller = style_controller

    def direct(self, storyboard: List[Dict[str, str]]) -> List[Dict[str, str]]:
        directed = []
        for shot in storyboard:
            shot = dict(shot)
            shot["director_notes"] = "强调角色目光与场景纵深，保持电影化构图。"
            shot["style_prompt"] = self.style_controller.apply(shot["visual"])
            directed.append(shot)
        return directed
