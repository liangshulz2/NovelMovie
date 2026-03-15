from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class OllamaConfig:
    model: str = "qwen2.5:14b"
    host: str = "http://127.0.0.1:11434"
    temperature: float = 0.7


@dataclass
class ComfyUIConfig:
    host: str = "http://127.0.0.1:8188"
    workflow_path: str = "workflows/wan2.2_image.json"
    video_workflow_path: str = "workflows/wan2.2_video.json"


@dataclass
class AppConfig:
    project_name: str = "NovelMovie_AI_V5"
    input_novel_path: Path = Path("input/novel.txt")
    output_dir: Path = Path("output")
    fps: int = 24
    default_shot_seconds: float = 4.0
    style_preset: str = "cinematic, high-detail, dramatic lighting"
    language: str = "zh-CN"
    voice_name: str = "zh-CN-XiaoxiaoNeural"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    comfyui: ComfyUIConfig = field(default_factory=ComfyUIConfig)
    extra: Dict[str, str] = field(default_factory=dict)


CONFIG = AppConfig()
