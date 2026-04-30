# embeddings.py — wraps OllamaEmbeddings to apply nomic-embed-text's prefixes.
#
# nomic-embed-text is asymmetric: it expects the texts you index and the
# queries you search with to be marked differently. Without these prefixes
# retrieval quality drops noticeably (≈30% nDCG penalty on standard benchmarks).
#
#   • Documents: prepend "search_document: "
#   • Queries:   prepend "search_query: "
#
# We get this for free by subclassing the LangChain Embeddings interface.
# Anything downstream that takes a `Embeddings` (vector stores, retrievers,
# eval harnesses) sees the prefixed version transparently.

from langchain_ollama import OllamaEmbeddings


class NomicEmbeddings(OllamaEmbeddings):
    """OllamaEmbeddings with nomic-embed-text's search_*: prefixes."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"search_query: {text}")
