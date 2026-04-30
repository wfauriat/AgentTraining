# scripts/index_docs.py — build the LanceDB vector index from a markdown corpus.
#
# Pipeline:
#   1. Load corpus (markdown)
#   2. MarkdownHeaderTextSplitter — split on H2 headings, attach heading metadata
#   3. RecursiveCharacterTextSplitter — sub-split any chunk over TARGET_CHUNK_CHARS
#   4. NomicEmbeddings — embed chunks (with search_document: prefix)
#   5. LanceDB — write {text, heading, vector} rows
#
# Run from lib_agent/:  python -m scripts.index_docs

from pathlib import Path

import lancedb
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from config import CORPUS_PATH, DB_PATH, EMBED_MODEL, TABLE_NAME, TARGET_CHUNK_CHARS
from embeddings import NomicEmbeddings


def main() -> None:
    md = Path(CORPUS_PATH).read_text(encoding="utf-8")
    print(f"Loaded {len(md):,} chars from {CORPUS_PATH}")

    # Step 1: split by H2. metadata["heading"] gets attached to each piece.
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "heading")],
        strip_headers=False,
    )
    docs = header_splitter.split_text(md)

    # Step 2: any chunk longer than TARGET_CHUNK_CHARS gets sub-split.
    # 200-char overlap mitigates context loss at chunk boundaries.
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TARGET_CHUNK_CHARS,
        chunk_overlap=200,
    )
    docs = sub_splitter.split_documents(docs)
    print(f"Split into {len(docs)} chunks")

    embedder = NomicEmbeddings(model=EMBED_MODEL)
    print(f"Embedding chunks (model: {EMBED_MODEL})...")
    vectors = embedder.embed_documents([d.page_content for d in docs])

    rows = [
        {
            "text": d.page_content,
            "heading": d.metadata.get("heading", "(top of document)"),
            "vector": v,
        }
        for d, v in zip(docs, vectors)
    ]

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
