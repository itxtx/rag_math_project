# src/config.py
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# Weaviate Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", None) # If you enable authentication

# Data Paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
RAW_LATEX_DIR = os.path.join(DATA_DIR, "raw_latex")
RAW_PDF_DIR = os.path.join(DATA_DIR, "raw_pdfs")
PROCESSED_TEXT_DIR = os.path.join(DATA_DIR, "processed_text")
PROCESSED_LATEX_DIR = os.path.join(PROCESSED_TEXT_DIR, "from_latex")
PROCESSED_PDF_DIR = os.path.join(PROCESSED_TEXT_DIR, "from_pdf")


# Ensure processed directories exist
os.makedirs(PROCESSED_LATEX_DIR, exist_ok=True)
os.makedirs(PROCESSED_PDF_DIR, exist_ok=True)

# Embedding Model (example, we'll refine this)
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Configuration (example)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(f"Weaviate URL: {WEAVIATE_URL}")
print(f"Raw LaTeX Dir: {RAW_LATEX_DIR}")
print(f"Raw PDF Dir: {RAW_PDF_DIR}")