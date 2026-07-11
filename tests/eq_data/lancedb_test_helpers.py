"""Shared test-only helper for registering fake LanceDB embedding functions
across the EQ RAG test suite -- avoids downloading a real model in the fast
unit test suite. Each call registers a uniquely-named embedding function
(via an incrementing counter) so multiple test files/fixtures never collide
on the same registry key.
"""
import itertools

from lancedb.embeddings import EmbeddingFunction, EmbeddingFunctionRegistry

_registry = EmbeddingFunctionRegistry.get_instance()
_counter = itertools.count()


def make_fake_embedding_func(vector_by_text, ndims=2):
    name = f"fake_eq_test_{next(_counter)}"

    @_registry.register(name)
    class _FakeEmbedder(EmbeddingFunction):
        def ndims(self):
            return ndims

        def compute_query_embeddings(self, query, *args, **kwargs):
            return [vector_by_text[str(query)]]

        def compute_source_embeddings(self, texts, *args, **kwargs):
            return [vector_by_text[str(t)] for t in texts]

    return _registry.get(name).create()
