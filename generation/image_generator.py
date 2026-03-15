from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from utils.file_utils import write_text


class ImageGenerator:
    def __init__(self, comfyui_host: str) -> None:
        self.comfyui_host = comfyui_host

    def generate(self, timeline: List[Dict[str, str]], output_dir: Path) -> List[Path]:
        image_paths: List[Path] = []
        for shot in timeline:
            shot_id = shot["shot_id"]
            path = output_dir / "frames" / f"{shot_id}.txt"
            write_text(path, f"[Mock image generated via ComfyUI@{self.comfyui_host}]\n{shot['style_prompt']}")
            image_paths.append(path)
        return image_paths
