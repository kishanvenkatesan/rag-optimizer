import json
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class QAExample:
    id: str
    query: str
    answer: str
    relevant_doc_ids: Optional[List[int]] = None

def load_qa_dataset(path: str) -> List[QAExample]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            examples.append(QAExample(
                id=str(obj["id"]),
                query=obj["query"],
                answer=obj["answer"],
                relevant_doc_ids=obj.get("relevant_doc_ids", []),
            ))
    return examples

def load_docs_corpus(path: str) -> List[str]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            docs.append(obj["text"])
    return docs
