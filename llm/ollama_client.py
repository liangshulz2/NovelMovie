from __future__ import annotations

import json
from typing import Any, Dict, Optional
from urllib import error, request


class OllamaClient:
    def __init__(self, host: str, model: str, temperature: float = 0.7, timeout: int = 120) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if system:
            payload["system"] = system

        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                parsed = json.loads(body)
                return parsed.get("response", "").strip()
        except (error.URLError, json.JSONDecodeError, TimeoutError):
            return ""

    def generate_json(self, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
        raw = self.generate(prompt, system=system)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
