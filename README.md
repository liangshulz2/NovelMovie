# NovelMovie_AI_V5

用 Python 串联 **Ollama + Wan2.2(ComfyUI 工作流) + AI 语音/配乐 + 自动剪辑** 的小说转动画电影流程。

## 流程
小说 -> 剧情解析 -> AI导演 -> 电影分镜 -> 角色一致性 -> 画面生成 -> 视频生成 -> AI配音 -> AI配乐 -> 自动剪辑 -> AI动画电影

## 运行
```bash
python3 main.py
```

> 默认读取 `input/novel.txt`，输出到 `output/`。

## 目录
- `llm/`: Ollama 客户端、剧情解析、分镜文案。
- `director/`: 导演策略、镜头规划、时间调度。
- `characters/`: 角色管理与一致性约束。
- `generation/`: 画面/视频/配音/配乐生成器（当前为可替换 mock 接口）。
- `postproduction/`: 自动剪辑与字幕。
- `pipeline/film_pipeline.py`: 全链路编排。

## 对接真实服务建议
- 在 `generation/image_generator.py` 与 `generation/video_generator.py` 中接入 ComfyUI API。
- 在 `generation/voice_generator.py` 接入 edge-tts / GPT-SoVITS。
- 在 `postproduction/video_editor.py` 改为调用 ffmpeg/moviepy。
