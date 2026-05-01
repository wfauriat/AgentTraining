# config.py — single place for constants. Mirrors agent/config.py.
MODEL = "qwen3:8b"
# Per-request context window passed to Ollama. Qwen3 8B is trained at 40K but
# Ollama's default is 4K. KV cache is ~70 MB / 1K tokens; 16K leaves ~1.5 GB
# headroom on an 8 GB GPU after Q4 weights (~5 GB). See `ollama ps` to verify.
NUM_CTX = 16384
WORKSPACE_DIR = "./workspace"

TAVILY_URL = "https://api.tavily.com/search"
MAX_TOOL_RESULT_CHARS = 4000   # truncate large tool outputs to keep context manageable
TOOL_TIMEOUT = 30              # wall-clock seconds for run_python

# RAG
EMBED_MODEL = "nomic-embed-text"
DB_PATH = "./vector_db"
TABLE_NAME = "course"
CORPUS_PATH = "./corpus/biology.md"
TARGET_CHUNK_CHARS = 2000
TOP_K = 5

# Context window management
# When the message list exceeds PRUNE_THRESHOLD_TOKENS, the prune node
# summarizes everything before the most recent HumanMessage boundary and
# replaces it with a stored summary in state["summary"].
PRUNE_THRESHOLD_TOKENS = 8000   # leave headroom under NUM_CTX (16384) for response + tool schemas
SUMMARY_KEEP_TAIL = 6           # try to keep last N messages intact (boundary-aware)
SUMMARY_MODEL = "qwen3-nothink" # fast, no reasoning tokens — well-suited to summarization
