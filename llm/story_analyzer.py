from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from utils.text_utils import chunk_text


@dataclass
class StoryAnalysis:
    summary: str
    characters: List[Dict[str, str]]
    arcs: List[str]
    scenes: List[Dict[str, str]]


class StoryAnalyzer:
    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    def analyze(self, novel_text: str) -> StoryAnalysis:
        chunks = chunk_text(novel_text, max_chars=1500)
        scenes = []
        for idx, chunk in enumerate(chunks, 1):
            scenes.append(
                {
                    "scene_id": f"S{idx:03d}",
                    "description": chunk[:300],
                    "emotion": "dramatic",
                    "location": "待推断",
                    "time": "未知",
                }
            )

        summary_prompt = f"请把以下小说内容总结成200字内剧情梗概:\n{novel_text[:3000]}"
        summary = self.llm_client.generate(summary_prompt) or "自动总结失败，使用规则摘要。"

        char_prompt = "从故事中识别主要角色，返回JSON列表，每项包含name,age,trait,goal。"
        llm_chars = self.llm_client.generate_json(char_prompt)
        characters = llm_chars.get("characters", []) if isinstance(llm_chars, dict) else []
        if not characters:
            characters = [{"name": "主角", "age": "未知", "trait": "坚韧", "goal": "完成使命"}]

        arcs = [
            "开端：人物与世界观建立",
            "发展：冲突升级与角色抉择",
            "高潮：关键对抗",
            "结局：主题回收与情感落点",
        ]

        return StoryAnalysis(summary=summary, characters=characters, arcs=arcs, scenes=scenes)
