MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"

MAX_TURNS = 8                   # agent loop ceiling
TOOL_TIMEOUT = 30               # seconds per tool call
MODEL_TIMEOUT = 120             # seconds per model call (first call loads weights)
MAX_TOOL_RESULT_TOKENS = 4000   # truncate large tool outputs
MAX_CONTEXT_TOKENS = 16000      # leave headroom on model window

TOOL_TIMEOUT = 30

TAVILY_URL = "https://api.tavily.com/search"

WORKSPACE_DIR = "./workspace"   # filesystem sandbox root


DB_PATH = "./vector_db"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

CORPUS_PATH = "./corpus/biology.md"
TABLE_NAME = "course"
TARGET_CHUNK_CHARS = 2000
TOP_K = 5


DEBUG = False
