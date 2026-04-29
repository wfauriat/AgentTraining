# scripts/index_docs.py
import re
import httpx
import lancedb

import numpy as np

from tools.embedding import embed_text

from config import CORPUS_PATH, DB_PATH, TABLE_NAME, EMBED_MODEL, OLLAMA_EMBED_URL, TARGET_CHUNK_CHARS

def load_markdown(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chunks(markdown: str) -> list[dict]:
    """
    Split markdown into chunks bounded by H2 headings.
    Sub-splits any chunk that exceeds TARGET_CHUNK_CHARS.
    Respects code fences (won't split on # inside ```...```).
    """
    chunks = []
    current_lines = []
    current_heading = "(top of document)"
    in_code_fence = False

    def emit():
        if not current_lines:
            return
        text = "\n".join(current_lines)
        if len(text) > TARGET_CHUNK_CHARS:
            for sub in sub_split(text):
                if len(sub) >= 100:  # filter out junk
                    chunks.append({"text": sub, "heading": current_heading})
        elif len(text) >= 100:  # also filter the non-subsplit case
            chunks.append({"text": text, "heading": current_heading})
        current_lines.clear()

    for line in markdown.splitlines():
        # Toggle code fence first — this protects against `#` inside code blocks
        if line.startswith("```"):
            in_code_fence = not in_code_fence

        # H2 heading outside a code fence = chunk boundary
        is_heading = line.startswith("## ") and not in_code_fence

        if is_heading:
            emit()
            current_heading = line.lstrip("# ").strip()

        current_lines.append(line)

    emit()  # don't forget the last chunk
    return chunks

def sub_split(text: str) -> list[str]:
    """Fallback splitter for over-long sections."""
    MIN_CHUNK_CHARS = 100  # anything shorter is treated as preamble

    parts = re.split(r"\n(?=### )", text)
    if len(parts) > 1:
        # Merge any tiny piece into the next one (typical: the H2 preamble)
        merged = []
        buffer = ""
        for p in parts:
            if len(p) < MIN_CHUNK_CHARS:
                buffer += p + "\n"
            else:
                merged.append(buffer + p)
                buffer = ""
        if buffer and merged:
            merged[-1] += "\n" + buffer
        elif buffer:
            merged.append(buffer)
        return merged

    # Paragraph fallback (unchanged)
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) > TARGET_CHUNK_CHARS and current:
            chunks.append(current.strip())
            current = p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current.strip():
        chunks.append(current.strip())
    return chunks


def main():
    print(f"Loading {CORPUS_PATH}...")
    md = load_markdown(CORPUS_PATH)

    print("Chunking...")
    chunks = split_into_chunks(md)
    print(f"  -> {len(chunks)} chunks")

    print(f"Embedding chunks (model: {EMBED_MODEL})...")
    rows = []
    for i, chunk in enumerate(chunks):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  {i+1}/{len(chunks)}: {chunk['heading'][:60]}")

        text_for_embedding = f"search_document: {chunk['heading']}\n\n{chunk['text']}"
        rows.append({
            "text": chunk["text"],          # store the body for display
            "heading": chunk["heading"],
            "vector": embed_text(text_for_embedding),  # but embed body+heading
        })

    print(f"Writing to {DB_PATH}/{TABLE_NAME}...")
    db = lancedb.connect(DB_PATH)
    try:
        db.drop_table(TABLE_NAME)
    except Exception:
        pass  # table didn't exist, fine
    db.create_table(TABLE_NAME, data=rows)

    print(f"Done. Indexed {len(rows)} chunks.")

if __name__ == "__main__":
    main()