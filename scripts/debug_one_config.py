import json
from pathlib import Path
from autoreg.data import load_qa_dataset, load_docs_corpus
from autoreg.rag_pipeline import RAGConfig, run_rag_experiment
from autoreg.chunking import build_chunk_corpus
from autoreg.retriever import SimpleEmbeddingModel, VectorRetriever
from autoreg.answering import AnswerGenerator, RAGSampleInput

INSPECT_CONFIG = RAGConfig(chunk_size=256, overlap=32, k=3)

def debug_run(dataset_path="data/sample.jsonl", docs_path="data/docs.jsonl"):
    dataset = load_qa_dataset(dataset_path)
    docs = load_docs_corpus(docs_path)

    chunk_texts, chunk_doc_ids = build_chunk_corpus(
        docs, chunk_size=INSPECT_CONFIG.chunk_size, overlap=INSPECT_CONFIG.overlap
    )

    print(f"TOTAL DOCS: {len(docs)}; TOTAL CHUNKS: {len(chunk_texts)}")

    embed_model = SimpleEmbeddingModel()
    retriever = VectorRetriever(chunk_texts, embed_model)

    answer_gen = AnswerGenerator()

    outputs = []
    for ex in dataset:
        retrieved = retriever.retrieve(ex.query, k=INSPECT_CONFIG.k)
        retrieved_chunk_indices = [idx for idx, _ in retrieved]
        retrieved_chunk_texts = [chunk_texts[idx] for idx in retrieved_chunk_indices]
        retrieved_doc_ids = [chunk_doc_ids[idx] for idx in retrieved_chunk_indices]

        sample = RAGSampleInput(query=ex.query, retrieved_docs=retrieved_chunk_texts)
        pred = answer_gen.generate_answer(sample)

        print("="*60)
        print(f"EXAMPLE ID: {ex.id}")
        print("Query:", ex.query)
        print("Gold answer:", ex.answer)
        print("Retrieved chunk indices:", retrieved_chunk_indices)
        print("Retrieved doc ids:", retrieved_doc_ids)
        print("Retrieved chunk texts (first 500 chars each):")
        for i, t in enumerate(retrieved_chunk_texts):
            print(f"  CHUNK {i} (len={len(t.split())} words): {t[:500]}")
        print("Predicted answer (first 500 chars):")
        print(pred[:500])
        print("="*60)

        outputs.append({
            "id": ex.id,
            "query": ex.query,
            "gold": ex.answer,
            "retrieved_chunk_indices": retrieved_chunk_indices,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieved_chunk_texts": retrieved_chunk_texts,
            "predicted": pred
        })

    out_path = Path("experiments/debug_one_config.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nWrote detailed debug results to {out_path.resolve()}")

if __name__ == "__main__":
    debug_run()
