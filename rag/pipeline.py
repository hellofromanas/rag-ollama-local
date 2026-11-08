"""The RAG pipeline: ingest -> retrieve -> build prompt -> generate."""
from __future__ import annotations

from pathlib import Path

from .chunker import Chunk, load_directory
from .embeddings import OllamaEmbedder, TfidfEmbedder
from .llm import OllamaLLM
from .store import VectorStore

PROMPT = """You are a careful assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly: "I don't know based on the provided documents."
Cite the sources you used in square brackets like [file.md#2].

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
    context = "\n\n".join(f"[{c.source}#{c.index}] {c.text}" for c, _ in hits)
    return PROMPT.format(context=context, question=question)


class RAG:
    def __init__(self, store: VectorStore, llm: OllamaLLM | None = None) -> None:
        self.store, self.llm = store, llm

    @classmethod
    def ingest(cls, docs_path: str | Path, index_dir: str | Path = "index", embedder: str = "tfidf",
               embed_model: str = "nomic-embed-text", size: int = 120, overlap: int = 30) -> "RAG":
        chunks = load_directory(docs_path, size, overlap)
        if not chunks:
            raise ValueError(f"No .md/.txt content found in {docs_path}")
        emb = TfidfEmbedder() if embedder == "tfidf" else OllamaEmbedder(embed_model)
        texts = [c.text for c in chunks]
        emb.fit(texts)
        store = VectorStore(emb.embed_documents(texts), chunks, emb)
        store.save(index_dir)
        return cls(store)

    @classmethod
    def load(cls, index_dir: str | Path = "index", llm: OllamaLLM | None = None) -> "RAG":
        return cls(VectorStore.load(index_dir), llm)

    def retrieve(self, question: str, k: int = 4) -> list[tuple[Chunk, float]]:
        return self.store.search(self.store.embedder.embed_query(question), k)

    def ask(self, question: str, k: int = 4) -> tuple[str, list[tuple[Chunk, float]]]:
        if self.llm is None:
            raise RuntimeError("No LLM configured")
        hits = self.retrieve(question, k)
        return self.llm.generate(build_prompt(question, hits)), hits
