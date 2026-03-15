from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from utils.file_utils import write_text


class SubtitleGenerator:
    def generate(self, timeline: List[Dict[str, str]], output_dir: Path) -> Path:
        lines = []
        for idx, shot in enumerate(timeline, 1):
            start = self._to_srt_time(float(shot["start"]))
            end = self._to_srt_time(float(shot["end"]))
            lines.append(str(idx))
            lines.append(f"{start} --> {end}")
            lines.append(shot.get("visual", ""))
            lines.append("")
        srt_path = output_dir / "subtitles.srt"
        write_text(srt_path, "\n".join(lines))
        return srt_path

    @staticmethod
    def _to_srt_time(seconds: float) -> str:
        millis = int((seconds - int(seconds)) * 1000)
        total = int(seconds)
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02}:{m:02}:{s:02},{millis:03}"
