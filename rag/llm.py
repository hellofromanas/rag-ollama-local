"""Thin client for a local Ollama server's text generation endpoint."""
from __future__ import annotations

import os

import requests


class OllamaLLM:
    def __init__(self, model: str | None = None, host: str = "http://localhost:11434", timeout: int = 300) -> None:
        self.model = model or os.environ.get("RAG_LLM_MODEL", "llama3.2")
        self.host, self.timeout = host.rstrip("/"), timeout

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
                timeout=self.timeout,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host} with model '{self.model}'. "
                f"Start Ollama and run:  ollama pull {self.model}\n{e}"
            ) from e
        return r.json()["response"].strip()
