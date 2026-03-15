from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from utils.file_utils import write_text


class VoiceGenerator:
    def __init__(self, voice_name: str) -> None:
        self.voice_name = voice_name

    def generate(self, timeline: List[Dict[str, str]], output_dir: Path) -> Path:
        script = []
        for shot in timeline:
            script.append(f"[{shot['shot_id']}] {shot.get('visual', '')}")
        voice_file = output_dir / "audio" / "narration.wav"
        write_text(voice_file, f"Mock TTS({self.voice_name})\n" + "\n".join(script))
        return voice_file
