# tools/web.py — web search (Tavily) and web fetch (httpx + trafilatura).
# LangChain @tool style: schema and description derived from type hints +
# docstring; no manual JSON-schema export needed.

import os

import httpx
import trafilatura
from dotenv import load_dotenv
from langchain_core.tools import tool

from config import MAX_TOOL_RESULT_CHARS, TAVILY_URL

# .env lives next to this package's parent (lib_agent/.env). load_dotenv()
# walks up from cwd by default, which works when chat.py / agent.py is run
# from lib_agent/.
load_dotenv()
_TAVILY_KEY = os.getenv("TAVILY_API_KEY")


def _format_tavily_results(results: list[dict]) -> str:
    """Render Tavily's JSON results as numbered text the model can read."""
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r.get('title', '(no title)')}")
        lines.append(f"   {r.get('content', '')}")
        lines.append(f"   {r.get('url', '')}")
        lines.append("")
    return "\n".join(lines)


@tool
def web_search(query: str) -> str:
    """Search the web via Tavily. Returns the top 5 results, each with title,
    snippet, and URL. Use this for broad lookup; follow up with web_fetch when
    you have a specific URL to read.

    Args:
        query: search terms.
    """
    if not _TAVILY_KEY:
        return "[error: TAVILY_API_KEY missing in .env]"
    try:
        response = httpx.post(
            TAVILY_URL,
            json={"api_key": _TAVILY_KEY, "query": query, "max_results": 5},
            timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        return f"[error: web_search failed — {e}]"

    results = response.json().get("results", [])
    if not results:
        return "[web_search: no results found]"
    return _format_tavily_results(results)


@tool
def web_fetch(url: str) -> str:
    """Fetch a webpage and return its main text content (or raw JSON).
    Strips nav/ads/boilerplate via trafilatura. Truncates large pages.

    Args:
        url: full URL of the page to fetch.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as e:
        return f"[error: web_fetch failed — {e}]"

    content_type = response.headers.get("content-type", "")
    if "json" in content_type:
        return response.text[:MAX_TOOL_RESULT_CHARS]

    text = trafilatura.extract(response.text)
    if not text:
        return "[web_fetch: could not extract content from page]"
    return text[:MAX_TOOL_RESULT_CHARS]
