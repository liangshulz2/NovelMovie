from __future__ import annotations

from typing import Dict


class CharacterConsistency:
    def enforce_prompt(self, prompt: str, character_db: Dict[str, Dict[str, str]]) -> str:
        tokens = []
        for name, meta in character_db.items():
            trait = meta.get("trait", "distinctive")
            tokens.append(f"{name}({trait})")
        consistency_tag = ", ".join(tokens) if tokens else "generic characters"
        return f"{prompt}; keep character consistency: {consistency_tag}"
