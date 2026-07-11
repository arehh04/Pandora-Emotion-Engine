"""Branch-scoped native-hybrid retrieval (LanceDB) with optional
cross-encoder reranking over the EQ RAG corpus built by
src.eq_data.build_eq_corpus.
"""

EXEMPLAR_METADATA_COLUMNS = ["tier", "tier_label", "eq_proxy_score", "source", "emotion_labels", "valence", "arousal", "dominance"]
THEORY_METADATA_COLUMNS = ["id", "topic", "citation_needed"]


def _search(table, query_text, k, fetch_k, reranker, metadata_columns):
    if table.count_rows() == 0:
        return []

    query = table.search(query_text, query_type="hybrid").limit(fetch_k)
    if reranker is not None:
        query = query.rerank(reranker=reranker)
    df = query.limit(k).to_pandas()

    hits = []
    for _, row in df.iterrows():
        hit = {"text": row["text"]}
        for col in metadata_columns:
            if col in df.columns and row[col] == row[col]:  # excludes NaN (NaN != NaN)
                hit[col] = row[col]
        hits.append(hit)
    return hits


def retrieve_similar_eq_exemplars(query_text, branch, rag_ctx, k=5, fetch_k=50):
    table = rag_ctx["exemplar_tables"][branch]
    return _search(table, query_text, k, fetch_k, rag_ctx.get("reranker"), EXEMPLAR_METADATA_COLUMNS)


def retrieve_relevant_msc_theory(query_text, branch, rag_ctx, k=3, fetch_k=50):
    table = rag_ctx["theory_tables"][branch]
    return _search(table, query_text, k, fetch_k, rag_ctx.get("reranker"), THEORY_METADATA_COLUMNS)
