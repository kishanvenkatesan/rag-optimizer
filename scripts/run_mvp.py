from autoreg.data import load_qa_dataset, load_docs_corpus
from autoreg.optimizer import grid_search, two_stage_search

if __name__ == "__main__":
    dataset_path = "data/wiki_eval.jsonl"
    docs_path = "data/wiki_docs.jsonl"

    dataset = load_qa_dataset(dataset_path)
    docs_corpus = load_docs_corpus(docs_path)

    search_space = {
    "chunk_size": [128, 256, 512],
    "overlap": [32, 64],
    "k": [1, 3, 5],
    "use_reranker": [False, True],
    "top_n": 20
    }

    MAX_EVAL = 80 
    dataset = dataset[:MAX_EVAL]

    #results = grid_search(dataset, docs_corpus, search_space)
    results = two_stage_search(
    dataset,
    docs_corpus,
    search_space,
    stage1_max_eval=200,   # retrieval-only size
    stage2_max_eval=60,    # expensive generation/judge size
    top_n_configs=5,       # only run full RAG for top 5 configs
    )

    print("=== Config Results (sorted by judge_score, then f1, then recall) ===")
    print("chunk_size | overlap | k | avg_judge | avg_f1 | avg_recall | combined")
    for r in results:
        # decide combined metric (weighted)
        combined = 0.7 * getattr(r, "avg_judge_score", 0.0) + 0.2 * getattr(r, "avg_f1_answer", 0.0) + 0.1 * getattr(r, "avg_recall_at_k", 0.0)
        print(
            f"{r.config.chunk_size:10d} | "
            f"{r.config.overlap:7d} | "
            f"{r.config.k:1d} | "
            f"{getattr(r, 'avg_judge_score', 0.0):9.3f} | "
            f"{getattr(r, 'avg_f1_answer', 0.0):7.3f} | "
            f"{r.avg_recall_at_k:10.3f} | "
            f"{combined:8.3f}"
        )
