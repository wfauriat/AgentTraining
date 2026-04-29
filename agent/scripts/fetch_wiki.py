# scripts/fetch_wiki.py
import httpx
import os

topics = [
    "Photosynthesis",
    "Cellular respiration",
    "Mitochondrion",
    "DNA",
    "RNA",
    "Ribosome",
    "Protein biosynthesis",
    "Eukaryote",
    "Prokaryote",
    "Cell biology",
]

os.makedirs("corpus", exist_ok=True)

with open("corpus/biology.md", "w", encoding="utf-8") as f:
    for topic in topics:
        # Wikipedia's "extracts" API returns plain text content
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "true",      # plain text, no HTML
            "titles": topic,
            "format": "json",
            "redirects": "1",           # follow redirects
        }
        response = httpx.get(
            url,
            params=params,
            headers={"User-Agent": "AgentTrainingBot/1.0 (educational use)"},
            timeout=30,
        )
        response.raise_for_status()
        
        pages = response.json()["query"]["pages"]
        page = next(iter(pages.values()))   # there's just one
        content = page.get("extract", "")
        
        if not content:
            print(f"  WARNING: no content for {topic}")
            continue
        
        f.write(f"## {topic}\n\n{content}\n\n")
        print(f"Got {topic}: {len(content)} chars")

print("\nDone. Wrote corpus/biology.md")