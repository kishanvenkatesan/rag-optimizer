from datasets import load_dataset
import json
from pathlib import Path

DATA_DIR = Path("data")
DOCS_OUT = DATA_DIR / "wiki_docs.jsonl"
EVAL_OUT = DATA_DIR / "wiki_eval.jsonl"

MAX_DOCS = 3000     # passages for corpus
MAX_QA = 500        # QA pairs for evaluation


def prepare_docs():
    print("Loading Wikipedia passages (corpus)...")
    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")

    split_name = list(ds.keys())[0]   # usually 'passages'
    dataset = ds[split_name]

    DATA_DIR.mkdir(exist_ok=True)

    written = 0
    with open(DOCS_OUT, "w", encoding="utf-8") as f:
        for ex in dataset:
            # text field can be 'passage' or similar
            text = ex.get("passage") or ex.get("text")
            if not text:
                continue

            obj = {
                "id": written,
                "text": text.strip()
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

            if written >= MAX_DOCS:
                break

    print(f"✅ Wrote {written} passages to {DOCS_OUT}")


def prepare_eval():
    print("Loading Wikipedia QA pairs (evaluation set)...")
    qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")

    split_name = list(qa.keys())[0]  
    dataset = qa[split_name]

    written = 0
    with open(EVAL_OUT, "w", encoding="utf-8") as f:
        for ex in dataset:
            question = ex.get("question")
            answer = ex.get("answer")

            if not question or not answer:
                continue

            # passage_id may or may not exist
            rel_ids = []
            if "passage_id" in ex and ex["passage_id"] is not None:
                try:
                    rel_ids = [int(ex["passage_id"])]
                except Exception:
                    rel_ids = []

            obj = {
                "id": written,
                "query": question.strip(),
                "answer": answer.strip(),
                "relevant_doc_ids": rel_ids
            }

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

            if written >= MAX_QA:
                break

    print(f"✅ Wrote {written} QA pairs to {EVAL_OUT}")


def main():
    prepare_docs()
    prepare_eval()
    print("\n🎉 Wikipedia corpus + evaluation set prepared successfully.")


if __name__ == "__main__":
    main()
