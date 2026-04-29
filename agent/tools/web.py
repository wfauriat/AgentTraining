# tools/web.py — web search and fetch tools
import os
import httpx
import trafilatura
from dotenv import load_dotenv

from config import TAVILY_URL, MAX_TOOL_RESULT_TOKENS

load_dotenv()
api_key = os.getenv("TAVILY_API_KEY")


def format_results(results: list) -> str:
    """Format Tavily results as readable text for the model."""
    lines = []
    for i, result in enumerate(results, start=1):
        lines.append(f"{i}. {result['title']}")
        lines.append(f"   {result['content']}")
        lines.append(f"   {result['url']}")
        lines.append("")
    return "\n".join(lines)


def web_search(query: str) -> str:
    """Search the web via Tavily and return the top results."""
    response = httpx.post(
        TAVILY_URL,
        json={
            "api_key": api_key,
            "query": query,
            "max_results": 5,
        },
        timeout=30,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        return "[web_search: no results found]"
    return format_results(results)


def web_fetch(url: str) -> str:
    """Fetch a URL and return its main text content (or raw JSON)."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = httpx.get(url, headers=headers, timeout=30)

    content_type = response.headers.get("content-type", "")
    if "json" in content_type:
        return response.text[:MAX_TOOL_RESULT_TOKENS]

    text = trafilatura.extract(response.text)
    if not text:
        return "[web_fetch: could not extract content from page]"
    return text[:MAX_TOOL_RESULT_TOKENS]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information on a topic. Returns the top 5 results with title, snippet, and URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch the content of a specific web page and return its cleaned text. Use when you have a known URL to read.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the webpage to fetch",
                    }
                },
                "required": ["url"],
            },
        },
    },
]


def dispatch(tool_name: str, arguments: dict) -> str:
    if tool_name == "web_search":
        query = arguments.get("query", "")
        if not query:
            return "[error: web_search requires a query argument]"
        return web_search(query)

    if tool_name == "web_fetch":
        url = arguments.get("url", "")
        if not url:
            return "[error: web_fetch requires a url argument]"
        return web_fetch(url)

    return f"[error: unknown tool '{tool_name}']"