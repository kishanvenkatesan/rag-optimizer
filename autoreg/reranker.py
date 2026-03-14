from typing import List, Tuple
from sentence_transformers import CrossEncoder
import numpy as np

class CrossEncoderReranker:
    """
    Cross-encoder reranker wrapper using sentence-transformers CrossEncoder.
    Model choices: 'cross-encoder/ms-marco-MiniLM-L-6-v2' (small, fastish).
    """

    _shared_model = None

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if CrossEncoderReranker._shared_model is None:
            CrossEncoderReranker._shared_model = CrossEncoder(model_name, device="cpu")
        self.model = CrossEncoderReranker._shared_model

    def rerank(self, query: str, candidate_texts: List[str]) -> List[Tuple[int, float]]:
        """
        Rerank candidate_texts given query.
        Returns list of (index_in_candidate_texts, score) sorted by score desc.
        """
        if not candidate_texts:
            return []

        # CrossEncoder accepts pairs: (query, candidate)
        pairs = [(query, c) for c in candidate_texts]
        scores = self.model.predict(pairs)  # numpy array of floats
        # Pair each score with its index and sort descending
        indexed_scores = list(enumerate(scores.tolist()))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores

    def rerank_topk(self, query: str, candidate_texts: List[str], k: int) -> List[int]:
        ranked = self.rerank(query, candidate_texts)
        topk = [idx for idx, _ in ranked[:k]]
        return topk
