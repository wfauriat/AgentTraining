# tools/docs.py — RAG over the local indexed corpus
import httpx
import lancedb
import numpy as np

# Same constants as the indexer — keep them aligned!
from config import DB_PATH, TABLE_NAME, EMBED_MODEL, OLLAMA_EMBED_URL, TOP_K, DEBUG

from tools.embedding import embed_text

# Connect once at module load — the table is reused across calls
_db = lancedb.connect(DB_PATH)
_table = _db.open_table(TABLE_NAME)


def search_documents(query: str) -> str:
    try:
        prefixed_query = f"search_query: {query}"
        query_vector = embed_text(prefixed_query)
        results = _table.search(query_vector).distance_type("cosine").limit(TOP_K).to_pandas()  # type: ignore[attr-defined]
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
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A natural-language question or topic to search for.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def dispatch(tool_name: str, arguments: dict) -> str:
    if tool_name == "search_documents":
        query = arguments.get("query", "")
        if not query:
            return "[error: search_documents requires a query argument]"
        return search_documents(query)

    return f"[error: unknown tool '{tool_name}']"