from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

# ---- Optimizer imports ----
from autoreg.data import load_qa_dataset, load_docs_corpus
from autoreg.optimizer import grid_search

# ---- RAG runtime imports ----
from autoreg.answering import AnswerGenerator, RAGSampleInput
from autoreg.chunking import build_chunk_corpus
from autoreg.retriever import SimpleEmbeddingModel, VectorRetriever

try:
    from autoreg.reranker import CrossEncoderReranker
except Exception:
    CrossEncoderReranker = None

app = FastAPI(title="RAG Optimizer API")

# ===============  GLOBAL RAG STATE  ==================

_docs: List[str] = []
_chunk_texts: List[str] = []
_chunk_doc_ids: List[int] = []
_retriever: Optional[VectorRetriever] = None
_reranker = None
_answer_gen: Optional[AnswerGenerator] = None
_current_docs_path: Optional[str] = None

DEFAULT_CHUNK_SIZE = 256
DEFAULT_OVERLAP = 64
DEFAULT_TOP_N = 20
DEFAULT_USE_RERANKER = True


def build_index_from_docs(docs_path: Path):
    global _docs, _chunk_texts, _chunk_doc_ids
    global _retriever, _reranker, _answer_gen
    global _current_docs_path

    print(f"[API] Loading corpus from {docs_path}")

    _docs = []
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            _docs.append(obj["text"])

    _chunk_texts, _chunk_doc_ids = build_chunk_corpus(
        _docs,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_OVERLAP,
    )

    embed_model = SimpleEmbeddingModel()
    _retriever = VectorRetriever(_chunk_texts, embed_model)

    if DEFAULT_USE_RERANKER and CrossEncoderReranker is not None:
        _reranker = CrossEncoderReranker()
    else:
        _reranker = None

    _answer_gen = AnswerGenerator(max_new_tokens=32, device=-1)

    _current_docs_path = str(docs_path)

    print(f"[API] Index built. Docs: {len(_docs)} | Chunks: {len(_chunk_texts)}")

# ================== STARTUP LOAD =====================

@app.on_event("startup")
def startup_init():
    default_path = Path("data/wiki_docs.jsonl")
    build_index_from_docs(default_path)

# ================== HEALTH CHECK =====================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "docs_path": _current_docs_path,
        "docs_loaded": len(_docs),
        "chunks": len(_chunk_texts),
    }

# ================== LIVE QUERY =======================

@app.get("/query")
def query_rag(
    q: str = Query(...),
    k: int = Query(3, ge=1, le=8),
    use_reranker: bool = Query(DEFAULT_USE_RERANKER),
    top_n: int = Query(DEFAULT_TOP_N),
):
    if _retriever is None:
        return {"error": "Retriever not initialized"}

    top_n = max(top_n, k)

    retrieved_topn = _retriever.retrieve(q, k=top_n)
    retrieved_topn_indices = [idx for idx, _ in retrieved_topn]
    retrieved_topn_texts = [_chunk_texts[idx] for idx in retrieved_topn_indices]

    reranker_scores = None
    final_chunk_indices = retrieved_topn_indices[:k]

    if use_reranker and _reranker and retrieved_topn_texts:
        ranked = _reranker.rerank(q, retrieved_topn_texts)
        topk_idx = [i for i, _ in ranked[:k]]
        final_chunk_indices = [retrieved_topn_indices[i] for i in topk_idx]
        reranker_scores = [float(score) for _, score in ranked[:k]]

    retrieved_chunks = [_chunk_texts[i] for i in final_chunk_indices]
    retrieved_doc_ids = [_chunk_doc_ids[i] for i in final_chunk_indices]

    sample = RAGSampleInput(query=q, retrieved_docs=retrieved_chunks)
    answer = _answer_gen.generate_answer(sample)

    return {
        "query": q,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "retrieved_doc_ids": retrieved_doc_ids,
        "reranker_scores": reranker_scores,
    }

# ================== RELOAD CORPUS ====================

class ReloadRequest(BaseModel):
    docs_path: str

@app.post("/reload_corpus")
def reload_corpus(req: ReloadRequest):
    docs_path = Path(req.docs_path)

    if not docs_path.exists():
        return {"status": "error", "message": "File does not exist"}

    build_index_from_docs(docs_path)

    return {
        "status": "ok",
        "docs_loaded": len(_docs),
        "chunks": len(_chunk_texts),
        "docs_path": str(docs_path),
    }

# ================== EXPERIMENTS ======================

class SearchSpace(BaseModel):
    chunk_size: List[int]
    overlap: List[int]
    k: List[int]

class ExperimentRequest(BaseModel):
    dataset_path: str
    docs_path: str
    search_space: SearchSpace

@app.post("/experiments")
def run_experiment(req: ExperimentRequest):
    dataset = load_qa_dataset(req.dataset_path)
    docs_corpus = load_docs_corpus(req.docs_path)

    results = grid_search(dataset, docs_corpus, req.search_space.dict())

    resp: List[Dict[str, Any]] = []
    for r in results:
        resp.append({
            "config": {
                "chunk_size": r.config.chunk_size,
                "overlap": r.config.overlap,
                "k": r.config.k,
            },
            "metrics": {
                "avg_f1_answer": r.avg_f1_answer,
                "avg_recall_at_k": r.avg_recall_at_k,
                "avg_judge_score": r.avg_judge_score,
            },
        })

    return {"results": resp}
