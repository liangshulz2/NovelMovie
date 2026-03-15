from __future__ import annotations

from pathlib import Path
from typing import List

from utils.file_utils import write_text


class VideoGenerator:
    def __init__(self, comfyui_host: str) -> None:
        self.comfyui_host = comfyui_host

    def generate_clips(self, frame_files: List[Path], output_dir: Path) -> List[Path]:
        clips: List[Path] = []
        for frame in frame_files:
            clip_path = output_dir / "clips" / f"{frame.stem}.mp4"
            write_text(clip_path, f"Mock video clip from {frame.name} via Wan2.2 @ {self.comfyui_host}")
            clips.append(clip_path)
        return clips
