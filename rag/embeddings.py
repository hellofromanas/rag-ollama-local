"""Embedders turn text into vectors. Two interchangeable implementations:

* TfidfEmbedder  - pure NumPy, no model download, keyword-based. Great for tests and a zero-setup demo.
* OllamaEmbedder - semantic embeddings from a local Ollama model (default: nomic-embed-text).
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import requests

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be by for from has have in is it its of on or that the this to was were will with".split())


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class TfidfEmbedder:
    kind = "tfidf"

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0, dtype=np.float32)

    def fit(self, docs: list[str]) -> "TfidfEmbedder":
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(_tokens(d)))
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in self.vocab], dtype=np.float32)
        return self

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(len(self.vocab), dtype=np.float32)
        for t, c in Counter(_tokens(text)).items():
            i = self.vocab.get(t)
            if i is not None:
                v[i] = (1 + math.log(c)) * self.idf[i]
        return v

    def embed_documents(self, docs: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vec(d) for d in docs])) if docs else np.zeros((0, len(self.vocab)), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vec(text)[None, :])[0]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "vocab": self.vocab, "idf": self.idf.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "TfidfEmbedder":
        e = cls()
        e.vocab = {k: int(v) for k, v in d["vocab"].items()}
        e.idf = np.array(d["idf"], dtype=np.float32)
        return e


class OllamaEmbedder:
    kind = "ollama"

    def __init__(self, model: str = "nomic-embed-text", host: str = "http://localhost:11434", timeout: int = 120) -> None:
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout

    def fit(self, docs: list[str]) -> "OllamaEmbedder":  # nothing to fit; kept for a uniform interface
        return self

    def _embed(self, texts: list[str]) -> np.ndarray:
        try:
            r = requests.post(f"{self.host}/api/embed", json={"model": self.model, "input": texts}, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Could not get embeddings from Ollama at {self.host} (model '{self.model}'). "
                f"Is Ollama running and the model pulled?  ollama pull {self.model}\n{e}"
            ) from e
        return _normalize(np.array(r.json()["embeddings"], dtype=np.float32))

    def embed_documents(self, docs: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(docs), 32):  # batch requests
            out.append(self._embed(docs[i : i + 32]))
        return np.vstack(out) if out else np.zeros((0, 0), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "model": self.model, "host": self.host}

    @classmethod
    def from_dict(cls, d: dict) -> "OllamaEmbedder":
        return cls(d["model"], d.get("host", "http://localhost:11434"))


def embedder_from_dict(d: dict):
    return {"tfidf": TfidfEmbedder, "ollama": OllamaEmbedder}[d["kind"]].from_dict(d)
