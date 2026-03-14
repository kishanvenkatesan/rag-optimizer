from typing import List, Set

def recall_at_k(retrieved_doc_ids: List[int], relevant_doc_ids: Set[int]) -> float:
    if not relevant_doc_ids:
        return 0.0
    hit = any(d in relevant_doc_ids for d in retrieved_doc_ids)
    return 1.0 if hit else 0.0

def mrr_at_k(retrieved_doc_ids: List[int], relevant_doc_ids: Set[int]) -> float:
    """
    Mean Reciprocal Rank: 1/rank of first relevant item, else 0.
    """
    if not relevant_doc_ids:
        return 0.0
    for rank, d in enumerate(retrieved_doc_ids, start=1):
        if d in relevant_doc_ids:
            return 1.0 / rank
    return 0.0

def f1_token_overlap(pred: str, gold: str) -> float:
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(set(pred_tokens))
    recall = len(common) / len(set(gold_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
