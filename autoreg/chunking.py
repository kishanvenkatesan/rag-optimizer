from typing import List, Tuple

def simple_chunk(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Safe chunking:
    - Guarantees forward progress.
    - If overlap >= chunk_size, we clamp it so step is at least 1.
    """
    words = text.split()
    if not words:
        return []

    if overlap >= chunk_size:
        # effectively no overlap or minimal
        overlap = chunk_size - 1 if chunk_size > 1 else 0

    step = max(1, chunk_size - overlap)

    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += step

    return chunks


def build_chunk_corpus(
    docs: List[str],
    chunk_size: int,
    overlap: int
) -> Tuple[List[str], List[int]]:
    """
    Returns:
      chunk_texts: list of chunk strings
      chunk_doc_ids: parallel list mapping chunk index -> original doc_id
    """
    chunk_texts: List[str] = []
    chunk_doc_ids: List[int] = []

    for doc_id, doc in enumerate(docs):
        chunks = simple_chunk(doc, chunk_size=chunk_size, overlap=overlap)
        for ch in chunks:
            chunk_texts.append(ch)
            chunk_doc_ids.append(doc_id)

    return chunk_texts, chunk_doc_ids
