# tools/docs.py — RAG search over the local LanceDB index.
# Uses NomicEmbeddings for the query side (with search_query: prefix); the
# index was built with the matching search_document: prefix in scripts/index_docs.py.

import lancedb
from langchain_core.tools import tool

from config import DB_PATH, EMBED_MODEL, TABLE_NAME, TOP_K
from embeddings import NomicEmbeddings

# Lazy-init: don't open the DB or load the embedder at import time. Means the
# agent starts even if the index doesn't exist yet — search just returns a
# clean error pointing at the indexer.
_embedder: NomicEmbeddings | None = None
_db = None
_table = None


def _ensure_ready():
    global _embedder, _db, _table
    if _embedder is None:
        _embedder = NomicEmbeddings(model=EMBED_MODEL)
    if _table is None:
        _db = lancedb.connect(DB_PATH)
        _table = _db.open_table(TABLE_NAME)
    return _embedder, _table


@tool
def search_documents(query: str) -> str:
    """Search the local indexed knowledge base by semantic similarity.
    Returns the top relevant text chunks, each tagged with its section heading.
    Use this for questions about topics covered in the local corpus.

    Args:
        query: a natural-language question or topic.
    """
    try:
        embedder, table = _ensure_ready()
    except Exception:
        return "[error: vector index not found — run `python -m scripts.index_docs` first]"

    try:
        vec = embedder.embed_query(query)
        results = (
            table.search(vec)
            .distance_type("cosine")  # type: ignore[attr-defined]
            .limit(TOP_K)
            .to_list()
        )
    except Exception as e:
        return f"[error: search_documents failed — {type(e).__name__}: {e}]"

    if not results:
        return "[search_documents: no results found]"

    lines: list[str] = []
    for i, row in enumerate(results, start=1):
        lines.append(f"--- Result {i} (from: {row['heading']}) ---")
        lines.append(row["text"])
        lines.append("")
    return "\n".join(lines)
