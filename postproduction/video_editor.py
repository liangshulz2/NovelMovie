from __future__ import annotations

from pathlib import Path
from typing import List

from utils.file_utils import write_text


class VideoEditor:
    def compose(self, clips: List[Path], narration: Path, bgm: Path, output_dir: Path) -> Path:
        final_path = output_dir / "final_movie.mp4"
        desc = ["Mock auto edit output", "Clips:"]
        desc.extend(str(c) for c in clips)
        desc.append(f"Narration: {narration}")
        desc.append(f"BGM: {bgm}")
        write_text(final_path, "\n".join(desc))
        return final_path
