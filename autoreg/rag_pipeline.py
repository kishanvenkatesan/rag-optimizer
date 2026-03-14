from dataclasses import dataclass
from typing import List

from .data import QAExample
from .retriever import VectorRetriever, SimpleEmbeddingModel
from .metrics import recall_at_k, mrr_at_k, f1_token_overlap
from .answering import AnswerGenerator, RAGSampleInput
from .chunking import build_chunk_corpus
from .judge import LLMJudge, JudgeInput
from .reranker import CrossEncoderReranker


@dataclass
class RAGConfig:
    chunk_size: int
    overlap: int
    k: int
    use_reranker: bool = False
    top_n: int = 20  # number of candidates to retrieve before reranking


@dataclass
class RAGResult:
    config: RAGConfig
    avg_recall_at_k: float
    avg_f1_answer: float
    avg_judge_score: float

@dataclass
class RetrievalResult:
    config: "RAGConfig"
    avg_recall_at_k: float
    avg_mrr_at_k: float

def run_rag_experiment(
    dataset: List[QAExample],
    config: RAGConfig,
    docs_corpus: List[str],
) -> RAGResult:
    """
    Runs one RAG experiment given a dataset, config, and raw docs.
    Now includes optional cross-encoder reranking.
    """

    # Build chunk-level corpus according to config
    chunk_texts, chunk_doc_ids = build_chunk_corpus(
        docs_corpus, chunk_size=config.chunk_size, overlap=config.overlap
    )

    # Shared embedding model (singleton) and retriever built on chunk_texts
    embed_model = SimpleEmbeddingModel()
    retriever = VectorRetriever(chunk_texts, embed_model)

    # Answer generator (LLM or local model)
    answer_gen = AnswerGenerator(device=0)

    # LLM-based judge (semantic scorer)
    judge = LLMJudge()

    # Optional cross-encoder reranker (shared)
    reranker = CrossEncoderReranker() if config.use_reranker else None

    recalls = []
    f1_scores = []
    judge_scores = []

    for ex in dataset:
        # Retrieve top-N candidate chunk indices (N = config.top_n)
        top_n = max(config.top_n, config.k)
        retrieved_topn = retriever.retrieve(ex.query, k=top_n)
        retrieved_topn_indices = [idx for idx, _ in retrieved_topn]
        retrieved_topn_texts = [chunk_texts[idx] for idx in retrieved_topn_indices]

        # If reranker enabled, rerank these candidate texts & select top-k
        if reranker is not None and retrieved_topn_texts:
            ranked = reranker.rerank(ex.query, retrieved_topn_texts)
            # ranked: list of (index_in_candidate_texts, score) sorted desc
            topk_indices_in_candidates = [i for i, _ in ranked[: config.k]]
            # map back to chunk_texts indices
            retrieved_chunk_indices = [retrieved_topn_indices[i] for i in topk_indices_in_candidates]
            # keep reranker scores if you want to surface them
        else:
            # no reranker: just take top-k from bi-encoder retrieval
            retrieved_chunk_indices = retrieved_topn_indices[: config.k]

        # Build retrieved texts and corresponding doc ids
        retrieved_chunk_texts = [chunk_texts[idx] for idx in retrieved_chunk_indices]
        retrieved_doc_ids = {chunk_doc_ids[idx] for idx in retrieved_chunk_indices}

        # Retrieval metric 
        relevant_indices = set(ex.relevant_doc_ids or [])
        recall_val = recall_at_k(list(retrieved_doc_ids), relevant_indices)
        recalls.append(recall_val)

        # Answer generation (use top retrieved chunks as context)
        sample = RAGSampleInput(query=ex.query, retrieved_docs=retrieved_chunk_texts)
        pred_answer = answer_gen.generate_answer(sample)

        # Lexical F1 (fallback metric)
        f1 = f1_token_overlap(pred_answer, ex.answer)
        f1_scores.append(f1)

        # LLM-based judge scoring (semantic, 0.0-1.0)
        context_concat = "\n\n".join(retrieved_chunk_texts[:5])  # top-5 chunks for judge
        judge_in = JudgeInput(
            query=ex.query,
            context=context_concat,
            predicted=pred_answer,
            gold=ex.answer,
        )
        try:
            jscore = judge.score(judge_in)
        except Exception:
            jscore = max(0.0, min(1.0, f1))
        judge_scores.append(jscore)

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0

    return RAGResult(
        config=config,
        avg_recall_at_k=avg_recall,
        avg_f1_answer=avg_f1,
        avg_judge_score=avg_judge,
    )


def run_retrieval_only_experiment(
    dataset: List["QAExample"],
    config: "RAGConfig",
    docs_corpus: List[str],
) -> RetrievalResult:
    """
    Stage-1: fast evaluation. Build chunks/index once per config and compute retrieval metrics only.
    """
    chunk_texts, chunk_doc_ids = build_chunk_corpus(
        docs_corpus, chunk_size=config.chunk_size, overlap=config.overlap
    )

    embed_model = SimpleEmbeddingModel()
    retriever = VectorRetriever(chunk_texts, embed_model)

    # optional reranker for stage-1 too 
    reranker = CrossEncoderReranker() if getattr(config, "use_reranker", False) else None

    recalls = []
    mrrs = []

    for ex in dataset:
        relevant = set(ex.relevant_doc_ids or [])
        if not relevant:
            # if dataset examples don’t include relevant ids, skip retrieval metrics
            continue

        top_n = max(getattr(config, "top_n", config.k), config.k)
        retrieved_topn = retriever.retrieve(ex.query, k=top_n)
        retrieved_topn_indices = [idx for idx, _ in retrieved_topn]
        retrieved_topn_texts = [chunk_texts[idx] for idx in retrieved_topn_indices]

        if reranker and retrieved_topn_texts:
            ranked = reranker.rerank(ex.query, retrieved_topn_texts)
            topk_idx_in_candidates = [i for i, _ in ranked[: config.k]]
            final_chunk_indices = [retrieved_topn_indices[i] for i in topk_idx_in_candidates]
        else:
            final_chunk_indices = retrieved_topn_indices[: config.k]

        retrieved_doc_ids = [chunk_doc_ids[idx] for idx in final_chunk_indices]

        recalls.append(recall_at_k(retrieved_doc_ids, relevant))
        mrrs.append(mrr_at_k(retrieved_doc_ids, relevant))

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0

    return RetrievalResult(config=config, avg_recall_at_k=avg_recall, avg_mrr_at_k=avg_mrr)
