# Starter Kit Spec: Local Agent on 8GB GPU

A minimal-but-complete specification for a personal agentic workflow running on consumer hardware. Kept deliberately light — enough to start building, not so much that it locks in decisions prematurely.

## 1. Target Stack

**Hardware**: NVIDIA GPU with 8GB VRAM (RTX 4060 / 4060 Ti / 3070 class), 16GB+ system RAM, ~20GB free disk for models.

**Runtime**: Ollama as the default. llama.cpp as the closer-to-metal alternative (more on the tradeoff below).

**Model**: Qwen 3.5 8B at Q4_K_M as primary; Gemma 4 E4B at Q4_K_M as fallback. Both pulled from the runtime's model registry.

**Language**: Python 3.11+. No framework dependency in v1 — hand-rolled loop using the runtime's HTTP API directly.

**Project layout**: Single repo, four files plus a tools directory. Resist the urge to scaffold more.

```
agent/
├── main.py          # entry point, CLI
├── loop.py          # agent loop, ~80 lines
├── tools/           # one file per tool
│   ├── web.py
│   ├── files.py
│   ├── python.py
│   ├── docs.py
│   └── meta.py
├── config.py        # model name, paths, limits
└── requirements.txt
```

## 2. Runtime Choice: Ollama vs llama.cpp

This is a real decision, so it's worth being explicit.

**Ollama** is a wrapper around llama.cpp with opinionated defaults, automatic model management, and a clean REST API. You run `ollama pull qwen3.5:8b-instruct-q4_K_M`, then POST to `localhost:11434/api/chat`. Tool calling is supported natively via a `tools` field that mirrors the OpenAI schema. Setup time: ten minutes. This is what 95% of local-agent projects should use.

**llama.cpp** is the underlying inference engine. Running it directly means compiling with the right CUDA flags, managing GGUF files yourself, and either using its built-in HTTP server (`llama-server`) or its Python bindings (`llama-cpp-python`). You get finer control over quantization variants, batch settings, KV cache configuration, sampling parameters, and GPU layer offloading. Tool calling support exists in `llama-server` via OpenAI-compatible endpoints, but is more recent and less polished than Ollama's.

**When to prefer llama.cpp**: you need a quantization Ollama doesn't ship (some IQ4_XS or imatrix variants squeeze quality out of 8GB), you're embedding the model inside a larger application, you want to tune sampling beyond what Ollama exposes, or you're hitting throughput limits and need fine control over batching and context shifting.

**Recommendation**: start with Ollama. Migrate to `llama-server` (still HTTP, still OpenAI-compatible) only if you hit a concrete limitation. The migration is mostly a base-URL change.

## 3. The Agent Loop

A single function, roughly this shape:

```
inputs:  user_message, tools[], max_turns
state:   conversation history (list of messages)
loop:
  send (history, tools) to model
  if response is text → return text, exit
  if response is tool_call(s):
    for each call:
      validate args against tool schema
      execute tool, capture result OR error
      append tool result to history
  increment turn counter
  if turn counter ≥ max_turns → force exit with truncation notice
```

Key design points: turn limit defaults to 8, every tool result is appended even on error (so the model can recover), and tool execution is wrapped in a timeout so a hung tool doesn't hang the whole agent. Validation uses Pydantic models per tool.

## 4. The Seven Starter Tools

Each tool is a Python function with a Pydantic input model and a docstring that becomes the model-facing description. The loop introspects these to build the tool schema sent to Ollama.

| Tool | Purpose | Backend / Library | Notes |
|---|---|---|---|
| `web_search` | Query a search engine | Tavily API (free tier) or Brave Search API | Returns top 5 results: title, snippet, URL |
| `web_fetch` | Retrieve and clean a URL | `httpx` + `trafilatura` | Truncate to ~4K tokens to protect context |
| `read_file` | Read a file from sandbox | stdlib | Path scoped to a `WORKSPACE_DIR` constant |
| `write_file` | Write a file to sandbox | stdlib | Same sandboxing; reject paths with `..` |
| `run_python` | Execute Python code | Docker container or E2B | 30-second timeout, no network by default |
| `search_documents` | RAG over local corpus | LanceDB or Chroma + a small embedding model | Returns chunks with doc IDs |
| `finish` | Explicit termination | No-op | Forces clean loop exit |

Tier-2 additions (`fetch_document`, `get_current_datetime`, `think`, `ask_human`) can be added once the seven above are working.

## 5. Configuration & Limits

Hard-coded as constants for v1; promote to env vars later.

- `MAX_TURNS = 8` — agent loop ceiling
- `TOOL_TIMEOUT = 30` (seconds) — per-call cap
- `MAX_TOOL_RESULT_TOKENS = 4000` — truncate large outputs before appending to history
- `MAX_CONTEXT_TOKENS = 16000` — leave headroom on the model's window
- `WORKSPACE_DIR = ./workspace` — filesystem sandbox root
- `MODEL = qwen3.5:8b-instruct-q4_K_M`

## 6. Embeddings for the RAG Tool

Run a small embedding model alongside the chat model. Ollama serves both from the same daemon, so this is one extra `ollama pull` and one extra HTTP endpoint. `nomic-embed-text` or `bge-small` are the standard picks — both fit in <500MB VRAM and won't meaningfully crowd out the chat model.

Indexing pipeline (one-time, separate script): chunk documents (~500 tokens, 50-token overlap) → embed → store in LanceDB. The `search_documents` tool just queries that index.

## 7. Observability (Bare Minimum)

Log every model call and every tool call to a JSONL file: timestamp, role, content, tool name, tool args, tool result, latency. This is non-negotiable — without it, debugging agent failures is guesswork. Twenty lines of code, pays for itself the first time the agent does something weird.

## 8. What's Deliberately Out of Scope for v1

- Multi-agent orchestration
- Streaming responses to the user
- Persistent memory across sessions
- Authentication, multi-user support
- Web UI (CLI is fine; add a UI when the loop is solid)
- Framework integration (LangChain, etc.)
- Hybrid local/API routing

Each of these is reasonable to add later. None should block getting a working v1.

## 9. Build Order

1. Get Ollama running, pull the model, verify a basic chat call works.
2. Write the loop with one trivial tool (`get_current_datetime`) to validate the plumbing.
3. Add `web_search` + `web_fetch`. Now the agent is genuinely useful.
4. Add `read_file` + `write_file` with sandboxing.
5. Add `run_python` in Docker. This is the fiddliest step; budget time for it.
6. Add the RAG pair (`search_documents`) once you have a corpus to index.
7. Add `finish` and tighten loop termination.

Each step is independently testable. Don't move forward until the previous step is reliable on at least five varied prompts.
