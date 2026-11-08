"""Split documents into overlapping word windows so each chunk fits an embedding model and keeps context."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str  # file name
    index: int   # chunk number within the file


def chunk_text(text: str, size: int = 120, overlap: int = 30) -> list[str]:
    """Return windows of `size` words that overlap by `overlap` words."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def load_directory(path: str | Path, size: int = 120, overlap: int = 30, exts=(".md", ".txt")) -> list[Chunk]:
    """Read every .md/.txt file under `path` and chunk it."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)
    out: list[Chunk] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        for i, piece in enumerate(chunk_text(text, size, overlap)):
            out.append(Chunk(piece, f.name, i))
    return out
