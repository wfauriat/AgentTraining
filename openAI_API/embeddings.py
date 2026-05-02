# embeddings.py — embeddings via the OpenAI-compatible /v1/embeddings endpoint.
#
# When EMBED_MODEL is "nomic-embed-text" we still need the asymmetric
# search_query: / search_document: prefixes — the model is asymmetric whether
# it's served by Ollama, vLLM, or TEI. Without them, retrieval quality drops
# noticeably (≈30% nDCG penalty on standard benchmarks). For other embedders
# (BGE, e5, etc.) the prefixes are different or unnecessary; toggle via the
# NOMIC_PREFIXES flag below.
#
# The wrapper subclasses OpenAIEmbeddings so anything downstream that takes
# a `Embeddings` (vector stores, retrievers) sees the prefixed version
# transparently.

from langchain_openai import OpenAIEmbeddings

from config import EMBED_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL

# Apply nomic-style prefixes only when the served model expects them.
NOMIC_PREFIXES = "nomic" in EMBED_MODEL.lower()


class _PrefixedOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAIEmbeddings with nomic-embed-text's search_*: prefixes."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if NOMIC_PREFIXES:
            texts = [f"search_document: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if NOMIC_PREFIXES:
            text = f"search_query: {text}"
        return super().embed_query(text)


# Single shared instance — module-level so callers can `from embeddings import embeddings`.
# check_embedding_ctx_length=False is required because OpenAIEmbeddings tries
# to use tiktoken to validate input length against the model's context window;
# for non-OpenAI backends (Ollama, vLLM, TEI) tiktoken's tokenizer is wrong
# for the actual model and the check produces spurious truncations.
embeddings = _PrefixedOpenAIEmbeddings(
    model=EMBED_MODEL,
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
    check_embedding_ctx_length=False,
)
