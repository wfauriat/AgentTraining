import os
import httpx
import trafilatura

from dotenv import load_dotenv

from config import TAVILY_URL


load_dotenv()  # reads .env file into environment variables
api_key = os.getenv("TAVILY_API_KEY")

def format_results(results: list) -> str:
    lines = []
    for i, result in enumerate(results, start=1):
        lines.append(f"{i}. {result['title']}")
        lines.append(f"   {result['content']}")
        lines.append(f"   {result['url']}")
        lines.append("")  # blank line between results
    return "\n".join(lines)

def web_search(query: str) -> str:
    response = httpx.post(TAVILY_URL,
                          json={
                              "api_key": api_key,
                              "query": query,
                              "max_results":5
                            }
                          )
    response.raise_for_status()
    results = response.json().get('results', [])
    if not results:
        return "[web_search: no results found]"
    return format_results(results)


# def web_fetch(url: str) -> str:
#     response = httpx.get(url)
#     response.raise_for_status()
#     payload = trafilatura.extract(response.content[:4000])
#     return payload if isinstance(payload, str) else  "No content"

def web_fetch(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = httpx.get(url, headers=headers, timeout=30)

    content_type = response.headers.get("content-type", "")
    if "json" in content_type:
        return response.text[:4000]
    else:
        text = trafilatura.extract(response.text)
        if not text:
            return "[web_fetch: could not extract content from page]"
        return text[:4000]
    

TOOLS = [

        {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search information only by fetching from a URL and return grounded content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    }
                },
                "required": ["query"]
            }
        }
    },
        {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch content of the web page and return curated text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The url of the webpage to fetch"
                    }
                },
                "required": ["url"]
            }
        }
    }
]

def dispatch(tool_name: str, arguments: dict) -> str:
    if tool_name == "web_search":
        query = arguments.get("query", "")
        if not query:
            return "[error: web_search requires a query argument]"
        return web_search(query)
    elif tool_name == "web_fetch":
        url = arguments.get("url", "")
        if not url:
            return "[error: web_fetch requires a url argument]"
        return web_fetch(url)