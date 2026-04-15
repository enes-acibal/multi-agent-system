import os
from dotenv import load_dotenv

load_dotenv() 

# ─── API Configuration ────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY is not set in the environment variables.")

# ─── Model Configuration ────
DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

# ─── RAG Configuration ────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512          # tokens per chunk
CHUNK_OVERLAP = 50        # overlap between chunks to preserve context
CHROMA_PERSIST_DIR = ".chroma"  # where ChromaDB stores vectors on disk
TOP_K_RESULTS = 5   # how many chunks to retrieve per query

# ─── Agent Configuration ────
MAX_AGENT_ITERATIONS = 10  # prevent infinite loops
AGENT_TIMEOUT = 60         # seconds per agent call