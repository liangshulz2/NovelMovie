from __future__ import annotations

from typing import Dict, List


class StoryboardWriter:
    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    def write_storyboard(self, scenes: List[Dict[str, str]]) -> List[Dict[str, str]]:
        boards: List[Dict[str, str]] = []
        for idx, scene in enumerate(scenes, 1):
            boards.append(
                {
                    "shot_id": f"SH{idx:03d}",
                    "scene_id": scene["scene_id"],
                    "visual": scene["description"],
                    "camera": "medium shot, slow dolly in",
                    "dialogue": "",
                    "duration": 4.0,
                }
            )
        return boards
