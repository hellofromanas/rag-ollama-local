"""Command line interface:  python -m rag ingest|ask|chat ..."""
from __future__ import annotations

import argparse
import sys

from .llm import OllamaLLM
from .pipeline import RAG


def _show_sources(hits) -> None:
    print("\nSources:")
    for chunk, score in hits:
        snippet = chunk.text[:110].replace("\n", " ")
        print(f"  [{chunk.source}#{chunk.index}]  score={score:.3f}  {snippet}...")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag", description="Local RAG with Ollama")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="chunk + embed a folder of .md/.txt files")
    ing.add_argument("docs", help="folder (or file) to index")
    ing.add_argument("--index", default="index")
    ing.add_argument("--embedder", choices=["tfidf", "ollama"], default="tfidf")
    ing.add_argument("--embed-model", default="nomic-embed-text")
    ing.add_argument("--chunk-size", type=int, default=120)
    ing.add_argument("--overlap", type=int, default=30)

    for name in ("ask", "chat"):
        s = sub.add_parser(name)
        s.add_argument("--index", default="index")
        s.add_argument("--k", type=int, default=4)
        s.add_argument("--model", default=None, help="Ollama chat model (default: $RAG_LLM_MODEL or llama3.2)")
        if name == "ask":
            s.add_argument("question")
            s.add_argument("--no-llm", action="store_true", help="show retrieved chunks only (no generation)")

    a = p.parse_args(argv)
    try:
        if a.cmd == "ingest":
            rag = RAG.ingest(a.docs, a.index, a.embedder, a.embed_model, a.chunk_size, a.overlap)
            print(f"Indexed {len(rag.store.chunks)} chunks with the '{a.embedder}' embedder into '{a.index}/'.")
        elif a.cmd == "ask":
            rag = RAG.load(a.index, None if a.no_llm else OllamaLLM(a.model))
            if a.no_llm:
                _show_sources(rag.retrieve(a.question, a.k))
            else:
                answer, hits = rag.ask(a.question, a.k)
                print(answer)
                _show_sources(hits)
        else:  # chat
            rag = RAG.load(a.index, OllamaLLM(a.model))
            print("Ask questions about your documents. Empty line to quit.")
            while (q := input("\n> ").strip()):
                answer, hits = rag.ask(q, a.k)
                print(answer)
                _show_sources(hits)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
