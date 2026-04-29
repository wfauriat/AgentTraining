# tools/embedding.py — shared embedding function
import httpx
import numpy as np
from config import EMBED_MODEL, OLLAMA_EMBED_URL


def embed_text(text: str) -> list[float]:
    """Get a normalized embedding from Ollama."""
    response = httpx.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    vec = np.array(response.json()["embedding"])
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()