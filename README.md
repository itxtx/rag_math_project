
```
rag_math_project/
├── .env                  # Environment variables (API keys, paths, etc.)
├── Dockerfile            # Dockerfile for your RAG application
├── docker-compose.yml    # Docker Compose to run Weaviate and your app
├── requirements.txt      # Python dependencies
├── data/
│   ├── raw_latex/        # Store your raw .tex files here
│   ├── raw_pdfs/         # Store PDF files (especially those without .tex source)
│   ├── processed_text/   # Standardized text/markdown output after parsing
│   │   ├── from_latex/
│   │   └── from_pdf/
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration (model names, paths, processing flags)
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   ├── document_loader.py # Detects file type and routes to appropriate parser
│   │   ├── latex_parser.py   # Parses .tex files, cleans, extracts math
│   │   ├── pdf_parser.py     # Parses .pdf files
│   │   │   ├── math_pdf_extractor.py # Specialized logic for mathematical PDFs (e.g., using Mathpix API, LlamaParse, etc.)
│   │   │   └── general_pdf_extractor.py # Standard PDF text extraction (e.g., PyMuPDF, pdf2embeddings)
│   │   ├── text_processor.py # Common text cleaning, structuring post-parsing
│   │   ├── chunker.py        # Text splitting/chunking strategies
│   │   └── vector_store_manager.py # Embedding generation and Weaviate indexing
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py      # Logic to query Weaviate
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm_handler.py    # Interface with the chosen LLM
│   │   └── prompt_template.py # Manage and format prompts
│   ├── pipeline.py         # Orchestrates the RAG flow (ingestion and query)
│   ├── api/                # (Optional) If building an API
│   │   ├── __init__.py
│   │   └── routes.py       # API endpoints
│   └── app.py              # Main application entry point
├── notebooks/
│   ├── 01_latex_parsing_experiment.ipynb
│   ├── 02_pdf_math_extraction_test.ipynb
│   ├── 03_chunking_strategy_evaluation.ipynb
│   ├── 04_embedding_retrieval_test.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_latex_parser.py
│   ├── test_pdf_parser.py
│   ├── test_chunker.py
│   └── test_retrieval.py
└── README.md
```