from __future__ import annotations

from typing import Dict, List


class SceneMemory:
    def __init__(self) -> None:
        self._memory: List[Dict[str, str]] = []

    def push(self, scene: Dict[str, str]) -> None:
        self._memory.append(scene)

    def recent(self, n: int = 3) -> List[Dict[str, str]]:
        return self._memory[-n:]
