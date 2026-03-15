from config import CONFIG
from llm.ollama_client import OllamaClient
from pipeline.film_pipeline import FilmPipeline


def main() -> None:
    llm_client = OllamaClient(
        host=CONFIG.ollama.host,
        model=CONFIG.ollama.model,
        temperature=CONFIG.ollama.temperature,
    )
    pipeline = FilmPipeline(CONFIG, llm_client)
    result = pipeline.run()
    print("NovelMovie pipeline completed")
    for k, v in result.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
