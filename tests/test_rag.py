"""Offline tests (no Ollama needed). Run:  python -m unittest discover -s tests -v"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chunker import chunk_text, load_directory  # noqa: E402
from rag.pipeline import RAG, build_prompt  # noqa: E402
from rag.store import VectorStore  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "docs" / "sample"


class ChunkerTests(unittest.TestCase):
    def test_overlap_and_coverage(self):
        words = [f"w{i}" for i in range(100)]
        chunks = chunk_text(" ".join(words), size=30, overlap=10)
        self.assertEqual(chunks[0].split()[-10:], chunks[1].split()[:10])  # overlap is preserved
        self.assertEqual(chunks[-1].split()[-1], "w99")                    # nothing is dropped
        self.assertTrue(all(len(c.split()) <= 30 for c in chunks))

    def test_short_and_empty(self):
        self.assertEqual(chunk_text("hello world", 30, 10), ["hello world"])
        self.assertEqual(chunk_text("   ", 30, 10), [])

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            chunk_text("a b c", size=5, overlap=5)


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.rag = RAG.ingest(SAMPLE, Path(cls.tmp.name) / "idx", embedder="tfidf", size=60, overlap=15)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def top_source(self, q):
        return self.rag.retrieve(q, k=1)[0][0].source

    def test_routes_questions_to_the_right_document(self):
        self.assertEqual(self.top_source("What is the Ollama API port and how do I pull a model?"), "ollama_basics.md")
        self.assertEqual(self.top_source("Why does overfitting happen and what is cross-validation?"), "model_evaluation.md")
        self.assertEqual(self.top_source("How are chunks embedded and retrieved with cosine similarity?"), "rag_basics.md")

    def test_scores_sorted_and_bounded(self):
        hits = self.rag.retrieve("embedding vectors", k=5)
        scores = [s for _, s in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(-1e-6 <= s <= 1 + 1e-6 for s in scores))

    def test_unrelated_query_scores_low(self):
        relevant = self.rag.retrieve("retrieval augmented generation", k=1)[0][1]
        unrelated = self.rag.retrieve("quarterly football fixtures banana", k=1)[0][1]
        self.assertLess(unrelated, relevant)


class StoreTests(unittest.TestCase):
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            rag = RAG.ingest(SAMPLE, Path(d) / "idx", embedder="tfidf")
            loaded = RAG.load(Path(d) / "idx")
            self.assertEqual(len(loaded.store.chunks), len(rag.store.chunks))
            self.assertTrue(np.allclose(loaded.store.vectors, rag.store.vectors))
            a = rag.retrieve("what is quantisation", 3)
            b = loaded.retrieve("what is quantisation", 3)
            self.assertEqual([c.index for c, _ in a], [c.index for c, _ in b])

    def test_missing_index(self):
        with self.assertRaises(FileNotFoundError):
            VectorStore.load("definitely-not-here")


class PromptTests(unittest.TestCase):
    def test_prompt_contains_question_context_and_citation_ids(self):
        with tempfile.TemporaryDirectory() as d:
            rag = RAG.ingest(SAMPLE, Path(d) / "idx")
            hits = rag.retrieve("what is RAG", 2)
            prompt = build_prompt("what is RAG", hits)
            self.assertIn("what is RAG", prompt)
            self.assertIn(f"[{hits[0][0].source}#{hits[0][0].index}]", prompt)
            self.assertIn("ONLY the context", prompt)


class LoaderTests(unittest.TestCase):
    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(load_directory(d), [])
            with self.assertRaises(ValueError):
                RAG.ingest(d, Path(d) / "idx")


if __name__ == "__main__":
    unittest.main()
