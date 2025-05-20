# src/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
# This is useful for local development to keep API keys out of the codebase.
# In a production/deployed environment, these would typically be set as actual environment variables.
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env') # Assuming .env is in project root
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# Weaviate Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
# WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY") # If using Weaviate Cloud Service with API key

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Data Directories (relative to project root, assuming src is in project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DATA_DIR_RAW_LATEX = os.path.join(DATA_DIR, "raw_latex")
DATA_DIR_RAW_PDFS = os.path.join(DATA_DIR, "raw_pdfs")

PARSED_CONTENT_DIR = os.path.join(DATA_DIR, "parsed_content")
DATA_DIR_PARSED_LATEX = os.path.join(PARSED_CONTENT_DIR, "from_latex")
DATA_DIR_PARSED_PDF = os.path.join(PARSED_CONTENT_DIR, "from_pdf")

# Mathpix API Configuration (Optional)
MATHPIX_APP_ID = os.getenv("MATHPIX_APP_ID")
MATHPIX_APP_KEY = os.getenv("MATHPIX_APP_KEY")

# LLM Configuration (Example for Gemini, adjust as needed)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Loaded from .env if present
QUESTION_GENERATION_LLM_MODEL = os.getenv("QUESTION_GENERATION_LLM_MODEL", "gemini-2.0-flash")
EVALUATION_LLM_MODEL = os.getenv("EVALUATION_LLM_MODEL", "gemini-2.0-flash")
# Search Defaults for Retriever
DEFAULT_SEARCH_LIMIT = int(os.getenv("DEFAULT_SEARCH_LIMIT", "5"))
DEFAULT_SEMANTIC_CERTAINTY = float(os.getenv("DEFAULT_SEMANTIC_CERTAINTY", "0.70"))
DEFAULT_HYBRID_ALPHA = float(os.getenv("DEFAULT_HYBRID_ALPHA", "0.5"))

# --- NEW: Persistence Configuration ---
# Path to the file that logs successfully processed document filenames.
# This file will be created in the DATA_DIR if it doesn't exist.
PROCESSED_DOCS_LOG_FILE = os.path.join(DATA_DIR, "processed_documents_log.txt")

# Ensure data directories exist (optional, can be handled by specific modules)
# os.makedirs(DATA_DIR_RAW_LATEX, exist_ok=True)
# os.makedirs(DATA_DIR_RAW_PDFS, exist_ok=True)
# os.makedirs(DATA_DIR_PARSED_LATEX, exist_ok=True)
# os.makedirs(DATA_DIR_PARSED_PDF, exist_ok=True)

# You can add other global configurations here
# For example, default chunk sizes, specific model parameters, etc.
