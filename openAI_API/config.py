# config.py — single place for constants. Mirrors lib_agent/config.py with
# the OpenAI-compatible endpoint swap applied.
import os

# ── Model + endpoint ──────────────────────────────────────────────────────
# Defaults target Ollama's OpenAI-compatible endpoint (served at /v1/ since
# Ollama 0.1.30) so this directory runs out-of-the-box on the same local rig
# the lib_agent build was developed on. For an air-gapped vLLM/llama.cpp
# deployment, override OPENAI_BASE_URL via env var (see docs/port_to_openai.md).
MODEL = os.getenv("LIB_AGENT_MODEL", "qwen3:8b")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
# vLLM accepts any non-empty string; Ollama ignores the key entirely.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")

# ── Workspace + tool config ───────────────────────────────────────────────
WORKSPACE_DIR = "./workspace"
TAVILY_URL = "https://api.tavily.com/search"
MAX_TOOL_RESULT_CHARS = 4000   # truncate large tool outputs to keep context manageable
TOOL_TIMEOUT = 30              # wall-clock seconds for run_python (Docker sandbox)
# Halved vs lib_agent — vLLM-style endpoints turn around tool selection in
# <2s and prose in <10s, so the prior 90s/180s budgets just hide hangs.
MODEL_TIMEOUT = 30             # httpx-level cap per ChatOpenAI call
TURN_TIMEOUT = 90              # wall-clock cap per REPL turn (excludes HITL waits)

# ── RAG ───────────────────────────────────────────────────────────────────
# Embeddings still hit the OpenAI-compatible endpoint via OpenAIEmbeddings.
# nomic-embed-text works through Ollama's /v1/embeddings; for other backends
# (vLLM, TEI), set EMBED_MODEL to whatever /v1/models reports for embeddings.
EMBED_MODEL = os.getenv("LIB_AGENT_EMBED_MODEL", "nomic-embed-text:latest")
DB_PATH = "./vector_db"
TABLE_NAME = "course"
CORPUS_PATH = "./corpus/biology.md"
TARGET_CHUNK_CHARS = 2000
TOP_K = 5

# ── Context window management ─────────────────────────────────────────────
# NUM_CTX is the per-request context window. Two consumers:
#   - chat.py status bar: denominator for the "ctx X/Y" percentage display
#   - Ollama runtime: forwarded via OLLAMA_EXTRA_BODY below so Ollama doesn't
#     silently fall back to its 4K default (visible via `ollama ps`)
# 16K is the lib_agent sweet spot for qwen3:8b on an 8 GB GPU — KV cache is
# ~70 MB / 1K tokens, leaving ~1.5 GB headroom after Q4 weights (~5 GB).
# Bump to 32K when running mistral-small3.2 (22B) or larger; drop to 8K if
# you see "out of memory" / GPU offload thrash.
NUM_CTX = int(os.getenv("LIB_AGENT_NUM_CTX", "16384"))
# When the message list exceeds PRUNE_THRESHOLD_TOKENS, the prune node
# summarizes everything before the most recent HumanMessage boundary and
# replaces it with a stored summary in state["summary"].
PRUNE_THRESHOLD_TOKENS = 8000
SUMMARY_KEEP_TAIL = 6           # try to keep last N messages intact (boundary-aware)
# Same model as MODEL: a single endpoint, no need to juggle two backends.
SUMMARY_MODEL = MODEL


# ── Ollama context-window note ────────────────────────────────────────────
# There is NO way to push num_ctx through Ollama's OpenAI-compat endpoint:
#   1. /v1/chat/completions silently drops the top-level `options` field, so
#      ChatOpenAI's extra_body={"options": {"num_ctx": N}} is a no-op.
#   2. A pre-flight /api/generate with options.num_ctx + keep_alive=-1 DOES
#      load the model at the right context, but the very first subsequent
#      /v1/chat/completions request (which has no num_ctx) reloads the model
#      back to Ollama's default 4K. The warmup gets clobbered.
# The fix is server-side or model-side, not client-side. Two options:
#
#   A. Server-wide default (one-time, affects all clients):
#        sudo systemctl edit ollama.service
#        # Add:  Environment="OLLAMA_CONTEXT_LENGTH=16384"
#        sudo systemctl restart ollama
#
#   B. Bake num_ctx into a custom model (one-time, affects only this model):
#        printf 'FROM qwen3:8b\nPARAMETER num_ctx 16384\n' \
#          | ollama create qwen3-16k -f -
#        # Then set MODEL=qwen3-16k (env or default below).
#
# Verify either fix worked with `ollama ps` — the CONTEXT column should
# match NUM_CTX after a single chat turn. Until one of them is applied,
# expect CONTEXT=4096 regardless of what's configured here.
