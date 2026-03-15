from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict

from characters.character_consistency import CharacterConsistency
from characters.character_manager import CharacterManager
from consistency.style_controller import StyleController
from consistency.temporal_consistency import TemporalConsistency
from director.camera_planner import CameraPlanner
from director.director_agent import DirectorAgent
from director.shot_scheduler import ShotScheduler
from generation.image_generator import ImageGenerator
from generation.music_generator import MusicGenerator
from generation.video_generator import VideoGenerator
from generation.voice_generator import VoiceGenerator
from llm.story_analyzer import StoryAnalyzer
from llm.storyboard_writer import StoryboardWriter
from postproduction.subtitle_generator import SubtitleGenerator
from postproduction.video_editor import VideoEditor
from utils.file_utils import ensure_dir, read_text, write_json
from world.scene_memory import SceneMemory
from world.world_state import WorldState


class FilmPipeline:
    def __init__(self, config, llm_client) -> None:
        self.config = config
        self.llm_client = llm_client

        self.story_analyzer = StoryAnalyzer(llm_client)
        self.storyboard_writer = StoryboardWriter(llm_client)
        self.style_controller = StyleController(config.style_preset)
        self.director_agent = DirectorAgent(self.style_controller)
        self.camera_planner = CameraPlanner()
        self.shot_scheduler = ShotScheduler()
        self.temporal_consistency = TemporalConsistency()
        self.character_manager = CharacterManager()
        self.character_consistency = CharacterConsistency()
        self.image_generator = ImageGenerator(config.comfyui.host)
        self.video_generator = VideoGenerator(config.comfyui.host)
        self.voice_generator = VoiceGenerator(config.voice_name)
        self.music_generator = MusicGenerator()
        self.video_editor = VideoEditor()
        self.subtitle_generator = SubtitleGenerator()
        self.scene_memory = SceneMemory()
        self.world_state = WorldState()

    def run(self) -> Dict[str, str]:
        ensure_dir(self.config.output_dir)
        novel_text = read_text(self.config.input_novel_path)

        analysis = self.story_analyzer.analyze(novel_text)
        self.character_manager.register(analysis.characters)

        boards = self.storyboard_writer.write_storyboard(analysis.scenes)
        directed = self.director_agent.direct(boards)

        for shot in directed:
            shot["style_prompt"] = self.character_consistency.enforce_prompt(
                shot["style_prompt"], self.character_manager.character_db
            )
            self.scene_memory.push({"shot_id": shot["shot_id"], "visual": shot["visual"]})

        camera_plans = self.camera_planner.plan(directed)
        timeline = self.shot_scheduler.schedule(camera_plans)
        timeline = self.temporal_consistency.smooth(timeline)

        frame_files = self.image_generator.generate(timeline, self.config.output_dir)
        clips = self.video_generator.generate_clips(frame_files, self.config.output_dir)
        narration = self.voice_generator.generate(timeline, self.config.output_dir)
        bgm = self.music_generator.generate(analysis.summary, self.config.output_dir)
        subtitles = self.subtitle_generator.generate(timeline, self.config.output_dir)
        final_movie = self.video_editor.compose(clips, narration, bgm, self.config.output_dir)

        manifest = {
            "summary": analysis.summary,
            "characters": analysis.characters,
            "arcs": analysis.arcs,
            "timeline_count": len(timeline),
            "world_state": asdict(self.world_state),
            "final_movie": str(final_movie),
            "subtitles": str(subtitles),
        }
        write_json(Path(self.config.output_dir) / "manifest.json", manifest)
        return {"final_movie": str(final_movie), "manifest": str(Path(self.config.output_dir) / "manifest.json")}
