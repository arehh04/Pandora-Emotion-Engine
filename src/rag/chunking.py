"""Plain sliding-window text chunking for the RAG corpus builder."""


def chunk_text(text, chunk_size=900, overlap=200):
    if not text:
        return []

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        chunks.append(text[start:start + chunk_size])
        if start + chunk_size >= n:
            break
        start += step
    return chunks
