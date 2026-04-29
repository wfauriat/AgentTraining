# tools/docs.py — RAG over the local indexed corpus
import lancedb
from pydantic import BaseModel, Field, ValidationError

from config import DB_PATH, TABLE_NAME, TOP_K, DEBUG
from tools.embedding import embed_text


# ── Argument schemas ───────────────────────────────────────────────────────

class SearchDocumentsArgs(BaseModel):
    query: str = Field(..., min_length=1, description="A natural-language question or topic to search for.")


# ── Lazy DB connection ─────────────────────────────────────────────────────
# Connect on first use, not at import time. This means the agent can start
# even if the index hasn't been built yet — search_documents just returns
# a clean error message.

_db = None
_table = None


def _get_table():
    global _db, _table
    if _table is None:
        _db = lancedb.connect(DB_PATH)
        _table = _db.open_table(TABLE_NAME)
    return _table


# ── Tool implementation ────────────────────────────────────────────────────

def search_documents(query: str) -> str:
    try:
        table = _get_table()
        prefixed_query = f"search_query: {query}"
        query_vector = embed_text(prefixed_query)
        results = (
            table.search(query_vector)
            .distance_type("cosine")  # type: ignore[attr-defined]
            .limit(TOP_K)
            .to_pandas()
        )
    except FileNotFoundError:
        return "[error: vector index not found — run scripts/index_docs.py first]"
    except Exception as e:
        return f"[error: search_documents failed — {e}]"

    if results.empty:
        return "[search_documents: no results found]"

    if DEBUG:
        print(f"  [debug] retrieved chunks for '{query}':")
        for i in range(len(results)):
            row = results.iloc[i]
            print(f"    {i+1}. {row['heading']} (distance: {row.get('_distance', '?'):.3f})")

    lines = []
    for i, (_, row) in enumerate(results.iterrows()):
        lines.append(f"--- Result {i+1} (from: {row['heading']}) ---")
        lines.append(row["text"])
        lines.append("")
    return "\n".join(lines)


# ── Schema sent to Ollama ──────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the local indexed knowledge base. "
                "Returns the top relevant text chunks with their section headings. "
                "Use this for questions about topics covered in the local corpus."
            ),
            "parameters": SearchDocumentsArgs.model_json_schema(),
        },
    }
]


# ── Dispatch ───────────────────────────────────────────────────────────────

def dispatch(tool_name: str, arguments: dict) -> str:
    if tool_name == "search_documents":
        try:
            args = SearchDocumentsArgs(**arguments)
        except ValidationError as e:
            return f"[error: invalid arguments for search_documents — {e}]"
        return search_documents(args.query)

    return f"[error: unknown tool '{tool_name}']"