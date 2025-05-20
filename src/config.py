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

# Output directory for parsed content (before chunking)
PARSED_OUTPUT_DIR = os.path.join(DATA_DIR, "parsed_content")
PARSED_LATEX_OUTPUT_DIR = os.path.join(PARSED_OUTPUT_DIR, "from_latex")
PARSED_PDF_OUTPUT_DIR = os.path.join(PARSED_OUTPUT_DIR, "from_pdf")

# Ensure output directories exist
os.makedirs(PARSED_LATEX_OUTPUT_DIR, exist_ok=True)
os.makedirs(PARSED_PDF_OUTPUT_DIR, exist_ok=True)

# API Keys for external services (example for Mathpix)
MATHPIX_APP_ID = os.getenv("MATHPIX_APP_ID")
MATHPIX_APP_KEY = os.getenv("MATHPIX_APP_KEY")

# Embedding Model (placeholder for now)
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Configuration (placeholder for now)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(f"Weaviate URL: {WEAVIATE_URL}")
print(f"Raw LaTeX Dir: {RAW_LATEX_DIR}")
print(f"Raw PDF Dir: {RAW_PDF_DIR}")
print(f"Parsed LaTeX Output Dir: {PARSED_LATEX_OUTPUT_DIR}")
print(f"Parsed PDF Output Dir: {PARSED_PDF_OUTPUT_DIR}")