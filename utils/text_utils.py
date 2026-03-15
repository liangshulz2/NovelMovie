from typing import Iterable, List


def chunk_text(text: str, max_chars: int = 1200) -> List[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    current = ""
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        candidate = f"{current}\n{paragraph}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def join_lines(lines: Iterable[str]) -> str:
    return "\n".join(line.strip() for line in lines if line.strip())
