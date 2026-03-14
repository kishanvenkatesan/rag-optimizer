from typing import List, Dict, Any
from .data import QAExample
from .rag_pipeline import RAGConfig, run_retrieval_only_experiment, run_rag_experiment, RAGResult

def grid_search(
    dataset: List[QAExample],
    docs_corpus: List[str],
    search_space: Dict[str, List[Any]],
) -> List[RAGResult]:
    results: List[RAGResult] = []

    for chunk_size in search_space["chunk_size"]:
        for overlap in search_space["overlap"]:
            if overlap >= chunk_size:
                continue
            for k in search_space["k"]:
                for use_reranker in search_space.get("use_reranker", [False]):
                    config = RAGConfig(
                        chunk_size=chunk_size,
                        overlap=overlap,
                        k=k,
                        use_reranker=use_reranker,
                        top_n=search_space.get("top_n", 20),
                    )
                    res = run_rag_experiment(dataset, config, docs_corpus)
                    results.append(res)

    # Sort primarily by avg_judge_score, then by avg_f1_answer, then recall
    results.sort(
        key=lambda r: (r.avg_judge_score, r.avg_f1_answer, r.avg_recall_at_k),
        reverse=True
    )
    return results

def two_stage_search(
    dataset: List[QAExample],
    docs_corpus: List[str],
    search_space: Dict[str, List[Any]],
    stage1_max_eval: int = 200,
    stage2_max_eval: int = 60,
    top_n_configs: int = 5,
) -> List[RAGResult]:
    """
    Stage 1: retrieval-only across many configs (fast).
    Stage 2: full RAG (generation + judge) only on top configs.
    """
    # ---- Stage 1 dataset slice ----
    ds1 = dataset[:stage1_max_eval]

    # Build candidate configs
    configs: List[RAGConfig] = []
    for chunk_size in search_space["chunk_size"]:
        for overlap in search_space["overlap"]:
            if overlap >= chunk_size:
                continue
            for k in search_space["k"]:
                for use_reranker in search_space.get("use_reranker", [False]):
                    cfg = RAGConfig(
                        chunk_size=chunk_size,
                        overlap=overlap,
                        k=k,
                        use_reranker=use_reranker,
                        top_n=search_space.get("top_n", 20),
                    )
                    configs.append(cfg)

    # ---- Stage 1: retrieval-only ----
    retrieval_results = []
    for cfg in configs:
        retrieval_results.append(run_retrieval_only_experiment(ds1, cfg, docs_corpus))

    # Rank configs by retrieval quality (MRR primary, recall secondary)
    retrieval_results.sort(key=lambda r: (r.avg_mrr_at_k, r.avg_recall_at_k), reverse=True)
    best_cfgs = [r.config for r in retrieval_results[:top_n_configs]]

    # ---- Stage 2 dataset slice ----
    ds2 = dataset[:stage2_max_eval]

    # ---- Stage 2: full RAG only on top configs ----
    final_results: List[RAGResult] = []
    for cfg in best_cfgs:
        final_results.append(run_rag_experiment(ds2, cfg, docs_corpus))

    # Sort by judge score (primary), then F1, then recall
    final_results.sort(key=lambda r: (r.avg_judge_score, r.avg_f1_answer, r.avg_recall_at_k), reverse=True)
    return final_results