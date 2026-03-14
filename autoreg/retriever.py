from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class SimpleEmbeddingModel:
    """
    Wrapper around a SentenceTransformer model.
    Uses a shared underlying model so we don't reload
    it for every config / experiment.
    """
    _shared_model = None

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if SimpleEmbeddingModel._shared_model is None:
            # Force CPU to avoid GPU OOM issues
            SimpleEmbeddingModel._shared_model = SentenceTransformer(
                model_name,
                device="cpu"
            )
        self.model = SimpleEmbeddingModel._shared_model

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True, 
        )
        return embeddings.astype("float32")


class VectorRetriever:
    def __init__(self, docs: List[str], embed_model: SimpleEmbeddingModel):
        self.docs = docs
        self.embed_model = embed_model
        self.index = None
        self._build_index()

    def _build_index(self):
        embeddings = self.embed_model.encode(self.docs)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[int, str]]:
        q_emb = self.embed_model.encode([query])
        scores, indices = self.index.search(q_emb, k)
        idxs = indices[0]
        return [(int(i), self.docs[int(i)]) for i in idxs]
