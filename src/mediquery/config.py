
"""Central configuration for MediQuery."""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
LOGS_DIR = PROJECT_ROOT / "logs"

SOURCE_PDF = RAW_DATA_DIR / "parcours_de_soins_diabete_type_2.pdf"

# ---------------------------------------------------------------------------
# Anthropic / Claude
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

CLAUDE_MODEL = os.getenv(
    "CLAUDE_MODEL",
    "claude-haiku-4-5-20251001",
)

CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "1500"))
CLAUDE_TEMPERATURE = float(os.getenv("CLAUDE_TEMPERATURE", "0.2"))

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# ---------------------------------------------------------------------------
# Document chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

VECTOR_SEARCH_K = int(os.getenv("VECTOR_SEARCH_K", "10"))
BM25_SEARCH_K = int(os.getenv("BM25_SEARCH_K", "10"))
HYBRID_SEARCH_K = int(os.getenv("HYBRID_SEARCH_K", "5"))

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

APP_TITLE = "MediQuery — Assistant médical"
APP_LANGUAGE = "fr"
