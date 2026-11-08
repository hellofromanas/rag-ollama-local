"""A tiny vector store: normalised vectors + metadata, cosine search with NumPy, persisted to disk."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .chunker import Chunk
from .embeddings import embedder_from_dict


class VectorStore:
    def __init__(self, vectors: np.ndarray, chunks: list[Chunk], embedder) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must have the same length")
        self.vectors, self.chunks, self.embedder = vectors, chunks, embedder

    def search(self, query_vec: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        """Top-k chunks by cosine similarity (vectors are L2-normalised, so a dot product is the cosine)."""
        if len(self.chunks) == 0:
            return []
        scores = self.vectors @ query_vec
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        (folder / "chunks.json").write_text(
            json.dumps([{"text": c.text, "source": c.source, "index": c.index} for c in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )
        (folder / "embedder.json").write_text(json.dumps(self.embedder.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, folder: str | Path) -> "VectorStore":
        folder = Path(folder)
        if not (folder / "vectors.npy").exists():
            raise FileNotFoundError(f"No index in {folder}. Run:  python -m rag ingest <docs folder>")
        vectors = np.load(folder / "vectors.npy")
        chunks = [Chunk(**c) for c in json.loads((folder / "chunks.json").read_text(encoding="utf-8"))]
        embedder = embedder_from_dict(json.loads((folder / "embedder.json").read_text(encoding="utf-8")))
        return cls(vectors, chunks, embedder)
