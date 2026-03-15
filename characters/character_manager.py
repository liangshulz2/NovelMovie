from __future__ import annotations

from typing import Dict, List


class CharacterManager:
    def __init__(self) -> None:
        self.character_db: Dict[str, Dict[str, str]] = {}

    def register(self, characters: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        for char in characters:
            name = char.get("name", "Unknown")
            self.character_db[name] = char
        return self.character_db
