# Local Agent Starter Kit

A minimal, hand-rolled agent that runs entirely on consumer hardware. Built as
a learning exercise: an LLM in a `while` loop, surrounded by seven tools, with
no framework dependencies.

The agent uses [Ollama](https://ollama.com) to host a local model and gives it
access to web search, a Python sandbox, file I/O scoped to a workspace
directory, and RAG over a local document corpus. Everything runs offline except
the optional web tools.

---

## Requirements

- **NVIDIA GPU with 8GB+ VRAM** (RTX 3070 / 4060 class). CPU-only works but is
  slow.
- **Python 3.10+** (uses `Path.is_relative_to`, the `int | None` syntax, etc.)
- **[Ollama](https://ollama.com)** installed and running
- **[Docker](https://docs.docker.com/get-docker/)** for the Python sandbox tool
- **Tavily API key** for web search ([free tier](https://tavily.com), 1000
  queries/month)

---

## Setup

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Pull the models Ollama needs
ollama pull qwen3:8b           # the chat model
ollama pull nomic-embed-text   # the embedding model for RAG

# 3. Pull the Docker image used by run_python
docker pull python:3.11-slim

# 4. Set up your API key
cp .env.example .env           # if you've created one; otherwise see below
echo "TAVILY_API_KEY=tvly-..." > .env

# 5. (Optional) Build the RAG index on a corpus you care about
python scripts/fetch_wiki.py        # builds ./corpus/biology.md as a demo
python -m scripts.index_docs        # indexes it into ./vector_db/
```

The `.env` file is gitignored. Never commit it.

---

## Running

From the `agent/` directory:

```bash
python main.py
```

You'll see:

```
Agent ready. Logging to logs/session_20260429_193044.jsonl. Type 'quit' to exit.
You:
```

Try prompts like:

- *"What time is it?"*
- *"Search the web for recent news on Mistral AI"*
- *"Use Python to compute the 100th Fibonacci number"*
- *"Save a note titled meeting.md with three bullet points about the project"*
- *"What does the corpus say about photosynthesis?"*

Type `quit` to exit cleanly.

---

## Architecture

```
agent/
├── main.py           # CLI entry point (input loop, history, prints)
├── loop.py           # the agent loop: send → branch on tool calls → repeat
├── observability.py  # JSONL session logger (one file per run)
├── config.py         # all constants in one place
├── tools/
│   ├── meta.py       # get_current_time, finish
│   ├── web.py        # web_search (Tavily), web_fetch (httpx + trafilatura)
│   ├── files.py      # read_file, write_file (sandboxed to ./workspace)
│   ├── python_sandbox.py   # run_python (Docker container, no network)
│   ├── docs.py       # search_documents (LanceDB + nomic-embed-text)
│   └── embedding.py  # shared Ollama embedding helper
├── scripts/
│   ├── fetch_wiki.py # demo corpus builder (Wikipedia API)
│   └── index_docs.py # build the LanceDB index from a markdown corpus
└── test_tools.py     # standalone test harness (run with `python test_tools.py`)
```

The loop itself is well under 100 lines. Every tool is one module: input
schema (Pydantic), implementation, public `TOOLS` list (sent to the model),
and a `dispatch` function. Adding a new tool is mechanical.

### The agent loop in plain English

```
for up to MAX_TURNS:
    send conversation history + tool list to the model
    if model returns tool calls:
        execute each tool, append results to history, loop
    if model calls finish(message):
        return message
    if model returns text:
        return text
return "stopped: max turns reached"
```

That's it. There is no clever orchestration.

---

## Configuration

All knobs live in `config.py`:

| Constant                  | Default                     | What it controls                          |
|---------------------------|-----------------------------|-------------------------------------------|
| `MODEL`                   | `qwen3:8b`                  | The Ollama chat model                      |
| `EMBED_MODEL`             | `nomic-embed-text`          | The embedding model for RAG                |
| `MAX_TURNS`               | `8`                         | Loop ceiling per user turn                 |
| `TOOL_TIMEOUT`            | `30` (s)                    | Per-tool execution cap                     |
| `MODEL_TIMEOUT`           | `120` (s)                   | Model HTTP call cap                        |
| `MAX_TOOL_RESULT_TOKENS`  | `4000`                      | Tool output truncation (rough char proxy)  |
| `WORKSPACE_DIR`           | `./workspace`               | Filesystem sandbox root                    |
| `DB_PATH`                 | `./vector_db`               | LanceDB location                           |
| `TABLE_NAME`              | `course`                    | LanceDB table name                         |
| `CORPUS_PATH`             | `./corpus/biology.md`       | Source corpus for indexing                 |
| `TARGET_CHUNK_CHARS`      | `2000`                      | Chunk size target during indexing          |
| `TOP_K`                   | `5`                         | Number of chunks returned per RAG query    |
| `DEBUG`                   | `False`                     | Print retrieved chunks during search       |

---

## Logging

Every session writes one JSONL file under `logs/`. Each line is one event:
user message, model call (with latency and tool call names), tool execution
(with arguments, result preview, and latency), agent reply.

To inspect a session quickly:

```bash
# Pretty-print events
cat logs/session_*.jsonl | jq

# Find slow tool calls
cat logs/session_*.jsonl | jq 'select(.event=="tool_call" and .latency_s > 5)'

# Count which tools the agent uses most
cat logs/session_*.jsonl | jq -r 'select(.event=="tool_call") | .tool' | sort | uniq -c
```

If something goes wrong and the agent does something weird, the log file is
the first place to look. It will save you guessing.

---

## Safety & sandboxing

This agent runs *real* code on your machine. The boundaries:

- **`read_file` / `write_file`** are confined to `WORKSPACE_DIR`. Path
  traversal (`../etc/passwd`, absolute paths, embedded `..` after a safe
  prefix, symlink trickery) is rejected by `safe_path`. Tested in
  `test_tools.py`.
- **`run_python`** runs in a Docker container with `--network none`,
  `--read-only`, capped memory and CPU, and a 30-second wall clock. The model
  cannot reach your filesystem, your environment variables, or the internet
  from inside this sandbox.
- **`web_fetch`** can read arbitrary URLs but only returns text. It does not
  execute JavaScript and cannot interact with login-walled content.
- **API keys** live in `.env`, which is gitignored.

These are reasonable defenses for a personal-use local agent. They are not a
substitute for not running the agent against fully untrusted inputs.

---

## Testing

```bash
python test_tools.py
```

Runs nine test cases covering: time stamping, Tavily result formatting, empty
search results, fetch truncation, dispatch error handling, path traversal
defense (multiple patterns), legitimate path acceptance, and the Python
sandbox returning expected output.

The test suite uses `unittest.mock.patch` to avoid hitting Tavily, the
network, or Docker during the run.

---

## Known limitations

- **RAG quality on technical corpora**: `nomic-embed-text` underperforms on
  code-heavy or terminology-dense content (e.g., the `LLM.md` course this was
  originally indexed against). Works well on prose corpora (Wikipedia, books).
  Try `mxbai-embed-large` or `bge-large` for technical material.
- **8B model size limits**: Qwen3 8B sometimes ignores tool descriptions —
  for example, replying with text instead of calling `finish` even when
  prompted to. The system prompt nudges this but doesn't enforce it.
- **No streaming**: replies appear all at once when the model is done. Easy
  to add later if needed.
- **No persistent memory**: each `python main.py` starts with an empty
  history.

---

## Build order

This repo was built incrementally following these steps. If you're learning
from it, walking the same path in your own checkout teaches more than reading
the finished code:

1. Make a single Ollama chat call from Python.
2. Wrap it in a loop with one trivial tool (`get_current_time`).
3. Add `web_search` + `web_fetch`.
4. Add `read_file` + `write_file` with sandboxing.
5. Add `run_python` in Docker.
6. Add the RAG pair (`search_documents`).
7. Add `finish` and tighten loop termination.

Plus extras: a test harness, observability, Pydantic argument validation,
heading-aware chunking, and a Wikipedia corpus fetcher.

---

## License

Personal learning project. Use however you like.