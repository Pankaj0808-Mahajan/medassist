"""Retriever: sentence embeddings + FAISS over the local guideline corpus (data/guidelines/*.md|txt|pdf)."""
import numpy as np

from .config import GUIDELINES

CHUNK, OVERLAP = 250, 50  # words (roughly 300-400 tokens)


def _read(path) -> str:
    if path.suffix == ".pdf":
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    return path.read_text(encoding="utf-8")


class Retriever:
    def __init__(self):
        import faiss
        from sentence_transformers import SentenceTransformer
        self.chunks = []
        for f in sorted(GUIDELINES.glob("*")):
            if f.suffix not in (".md", ".txt", ".pdf"):
                continue
            words = _read(f).split()
            for i in range(0, max(len(words), 1), CHUNK - OVERLAP):
                part = " ".join(words[i:i + CHUNK])
                if part:
                    self.chunks.append({"source": f.stem, "text": part})
                if i + CHUNK >= len(words):
                    break
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = None
        if self.chunks:
            emb = self.model.encode([c["text"] for c in self.chunks], normalize_embeddings=True).astype("float32")
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(emb)

    def search(self, query: str, k: int = 3) -> list:
        if self.index is None or not query.strip():
            return []
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [{"id": f"S{n}", **self.chunks[i], "score": float(s)} for n, (s, i) in enumerate(zip(scores[0], ids[0]), 1) if i >= 0]
