from __future__ import annotations

from pathlib import Path

from utils.file_utils import write_text


class MusicGenerator:
    def generate(self, summary: str, output_dir: Path) -> Path:
        music_file = output_dir / "audio" / "bgm.wav"
        write_text(music_file, f"Mock BGM generated from summary: {summary[:120]}")
        return music_file
