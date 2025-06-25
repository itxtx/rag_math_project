# Adaptive RAG-Based Learning System for Technical Content

## 1. Overview & Motivation

This project introduces an innovative adaptive learning system developed in Python, leveraging **Retrieval Augmented Generation (RAG)** to revolutionize how individuals engage with and learn from dense technical documents, particularly those formatted in LaTeX. Traditional learning from such materials can be static and overwhelming. This system aims to create a **dynamic, personalized learning journey** by:

* Intelligently ingesting and understanding complex technical content.
* Generating tailored questions to reinforce learning and assess comprehension.
* Adapting to the learner's knowledge state and focusing on areas needing improvement.
* (Ultimately) Incorporating principles like spaced repetition for long-term retention.

The core architecture processes technical documents, chunks them into meaningful semantic units, stores them in a Weaviate vector database, and uses this knowledge base with Large Language Models (LLMs) to power an interactive and adaptive learning experience.

## 2. Key Features

* **Advanced LaTeX Ingestion Pipeline:**
    * Robustly processes `.tex` files, including a pre-processing step for common custom LaTeX command expansion.
    * Parses content using `pylatexenc` for accurate structural understanding.
    * Identifies conceptual topics based on document hierarchy (sections, subsections).
    * Strategically chunks text using `langchain_text_splitters.RecursiveCharacterTextSplitter` for optimal retrieval.
* **Vector Knowledge Base:** Employs **Weaviate** to store text chunks and their **Sentence Transformers** embeddings, enabling powerful semantic search.
* **Intelligent Retrieval System:** Utilizes hybrid search strategies (semantic + keyword) to fetch the most relevant context for question generation and answer evaluation.
* **RAG-Powered Question Generation:** Leverages an LLM (e.g., Gemini Flash 2.0) to dynamically generate diverse questions (factual, conceptual, analytical) at varying difficulty levels, grounded in retrieved context.
* **LLM-Based Answer Evaluation:** Assesses learner responses against the source context using an LLM, providing quantitative scores and qualitative, constructive feedback.
* **Comprehensive Learner Profile Management:** Tracks learner progress, concept-specific knowledge scores, interaction history, and (future) Spaced Repetition System (SRS) data within an **SQLite** database.
* **Adaptive Learning Engine:**
    * Constructs a curriculum map from ingested documents.
    * Intelligently selects questions to target weak areas or introduce new concepts based on learner performance.
    * Adjusts question difficulty and the amount of context provided.
    * Allows learners to focus their study on specific documents/topics.
* **Spaced Repetition System (SRS):** Basic implementation to calculate next review dates for mastered concepts.
* **Scalable FastAPI Backend:** Exposes all system functionalities through a well-defined RESTful API, built for robustness and asynchronous operations.

## 3. System Architecture & Modules

The system features a modular design, primarily within the `src/` directory, promoting separation of concerns and maintainability:



* **data_ingestion/**: `latex_processor.py`, `latex_parser.py`, `pdf_parser.py`, `document_loader.py`, `concept_tagger.py`, `chunker.py`, `vector_store_manager.py`
* **retrieval/**: `retriever.py`
* **generation/**: `question_generator_rag.py`
* **evaluation/**: `answer_evaluator.py`
* **learner_model/**: `profile_manager.py`, `knowledge_tracker.py`
* **interaction/**: `answer_handler.py`
* **adaptive_engine/**: `question_selector.py`, `srs_scheduler.py`
* **api/**: `main_api.py`, `models.py`
* **pipeline.py**: Orchestrates key flows.
* **app.py**: Interactive CLI entry point.
* **config.py**: Centralized configuration management.

## 4. Technology Stack

This project utilizes a modern, robust technology stack:

* **Core Language:** Python 3.10+
* **Vector Database:** Weaviate (local Docker instance or cloud)
* **Embedding Models:** Sentence Transformers (e.g., `all-MiniLM-L6-v2`)
* **LaTeX Parsing:** `pylatexenc`
* **Text Chunking:** `langchain-text-splitters`
* **LLM Interaction:** Google Gemini API (leveraging `aiohttp` for efficient asynchronous calls)
* **API Framework:** FastAPI, Uvicorn
* **Data Validation:** Pydantic
* **Learner Profile Storage:** SQLite3
* **Asynchronous Programming:** `asyncio`
* **Configuration:** `python-dotenv` for `.env` file management
* **Testing Frameworks:** `unittest`, `unittest.mock`

## 5. Project Structure



```
rag_math_project/
├── data/
│   ├── raw_latex/
│   ├── raw_pdfs/
│   ├── parsed_content/
│   │   ├── from_latex/
│   │   └── from_pdf/
│   ├── learner_profiles.sqlite3
│   └── processed_documents_log.txt
├── src/
│   ├── adaptive_engine/
│   ├── api/
│   ├── data_ingestion/
│   ├── evaluation/
│   ├── generation/
│   ├── interaction/
│   ├── learner_model/
│   ├── retrieval/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   └── pipeline.py
├── tests/
│   # ... (module-specific test directories)
├── .env
├── requirements.txt
└── README.md
```

## 6. Setup and Installation

<<<<<<< test
=======

>>>>>>> main
* **Prerequisites:** Python 3.10+, Docker & Docker Compose, Weaviate access, LLM API Key. (Optional: Mathpix API credentials).
* **Installation:**
    1.  Clone: `git clone <your-repository-url> && cd rag_math_project`
    2.  Environment: `python -m venv .venv && source .venv/bin/activate` (or ` .venv\Scripts\activate` on Windows)
    3.  Install uv: `pip install uv`
    4.  Dependencies: `uv pip install .` (for development) or `uv pip install -e .` (for editable install)
    5.  Weaviate: Ensure instance is running (e.g., `docker-compose up -d` if using local compose file).
    6.  Environment Variables: Create `.env` in root with `GEMINI_API_KEY`, etc. (see `src/config.py` for all options).

## 7. Running the Application



* **a. Data Ingestion & Interactive CLI Demo:**
    * Place `.tex` files in `data/raw_latex/`.
    * Run: `python -m src.app`
* **b. API Server:**
    * Ensure data is ingested.
    * Run: `python -m src.api.main_api` or `uvicorn src.api.main_api:app --reload --host 0.0.0.0 --port 8000`
    * Access Swagger UI at `http://localhost:8000/docs`.
* **c. Running Tests:**
    * All: `python -m unittest discover tests`
    * Specific: `python -m unittest tests.module_name.test_file_name`

## 8. API Endpoints

The FastAPI backend provides a comprehensive set of endpoints. Key examples:

* `GET /api/v1/topics`: Lists available top-level learning topics.
* `POST /api/v1/interaction/start`: Initiates a learning session.
* `POST /api/v1/interaction/submit_answer`: Submits and evaluates a learner's answer.
* `GET /api/v1/health`: API health check.



## 9. Current Status, Challenges, and Limitations

This project is an actively developed prototype with a strong foundation. Current areas of focus and known limitations include:

* **Document Type Focus:** Primarily robust for LaTeX ingestion. PDF processing is currently basic and experimental.
* **Challenge: Advanced LaTeX Parsing:** While the system handles common custom LaTeX commands, ensuring reliable parsing for a diverse range of highly complex or obscure user-defined macros is an ongoing challenge. This can occasionally lead to incomplete or inaccurate content extraction.
* **Challenge: Semantic Chunking for Mathematical Content:** The current text chunking strategy (`RecursiveCharacterTextSplitter`) is general-purpose. For highly structured mathematical content, it can sometimes inadvertently separate semantically linked units (e.g., a theorem from its proof, or a definition from its explanatory examples). Refining chunking logic to be more context-aware for mathematical discourse is a key area for improvement.
* **Challenge: Mathematical PDF Extraction:** The existing PDF parser struggles with accurately extracting densely mathematical text, particularly complex equations and their surrounding layout. This significantly impacts the quality of ingested PDF content.
* **Database Scalability:** The current SQLite-based learner profile storage, while suitable for prototyping and single-user scenarios, would need to be migrated to a more robust database system (e.g., PostgreSQL) for production deployment. This is particularly important for handling concurrent write operations and ensuring data integrity in a multi-user environment.
* **Spaced Repetition System (SRS):** Implemented with basic logic; full SM-2 algorithm integration is planned.
* **Curriculum Structure:** Currently derived from document hierarchy. Explicit prerequisite definition between concepts is future work.
* **Error Handling & Logging:** Robust, but opportunities for further enhancement and more granular logging exist.
* **User Interface:** The interactive CLI (`app.py`) serves for demonstration and testing. A dedicated frontend application would consume the API for a richer user experience.

## 10. Future Work / Roadmap

The vision for this project includes several exciting enhancements:

* **Enhanced SRS:** Implement a full-fledged SM-2 algorithm with dynamic ease factors and review intervals.
* **Sophisticated Curriculum Graph:**
    * Allow manual definition and LLM-assisted inference of prerequisites between concepts.
    * Utilize graph traversal for more nuanced "adjacent possible" concept recommendations.
* **Advanced Question Generation:** Develop capabilities for more interactive question types, such as "fill-in-the-missing-step" for proofs or auto-generating cloze deletions from definitions.
* **Improved LaTeX & PDF Processing:**
    * Develop more resilient LaTeX pre-processing to handle a wider array of custom macros.
    * Investigate and integrate specialized OCR and document analysis tools for superior mathematical PDF parsing.
* **Context-Aware Semantic Chunking:** Research and implement advanced chunking strategies tailored for technical and mathematical documents to better preserve logical units.
* **Multi-Modal Content Support:** Extend ingestion to handle images, diagrams, and other embedded media within technical documents.
* **Dedicated Web Frontend:** Build a responsive web application for a seamless and engaging learner experience.
* **Performance Optimization:** Profile and optimize for large-scale knowledge bases and concurrent users.
* **Learner Analytics & Reporting:** Provide dashboards for learners and educators to track progress, identify challenging concepts, and assess system effectiveness
<<<<<<< test

# RAG Math Project

A Retrieval-Augmented Generation (RAG) system for mathematics education, designed to provide personalized learning experiences through adaptive question generation and concept tracking.

## Features

- **Adaptive Learning**: Personalized question generation based on learner's knowledge level
- **Concept Tracking**: Monitors learner progress across mathematical concepts
- **Curriculum Mapping**: Structured learning paths with concept dependencies
- **Latex Support**: Full support for mathematical notation and equations
- **Vector Search**: Efficient semantic search for relevant mathematical content
- **Knowledge Graph**: Tracks relationships between mathematical concepts

## Project Structure

```
rag_math_project/
├── src/
│   ├── data_ingestion/      # Data processing and ingestion
│   ├── knowledge_graph/     # Knowledge graph management
│   ├── learning/           # Learning session management
│   ├── question_gen/       # Question generation
│   ├── retrieval/          # Vector search and retrieval
│   └── utils/              # Utility functions
├── tests/                  # Test suite
├── data/                   # Data storage
└── config/                 # Configuration files
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-math-project.git
cd rag-math-project
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e ".[test]"
```

## Configuration

1. Create a `.env` file in the project root:
```bash
cp .env.example .env
```

2. Update the environment variables in `.env` with your configuration:
```env
WEAVIATE_URL=your_weaviate_url
WEAVIATE_API_KEY=your_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Start the FastAPI server:
```bash
uvicorn src.main:app --reload
```

2. Access the API documentation at `http://localhost:8000/docs`

## Testing

The project uses pytest for testing. To run the tests:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src

# Run specific test file
pytest tests/test_question_selector_pytest.py

# Run tests in parallel
pytest -n auto
```

## Development

### Code Style

The project follows PEP 8 guidelines. To check code style:

```bash
flake8 src tests
```

### Type Checking

Type hints are used throughout the project. To check types:

```bash
mypy src
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Weaviate](https://weaviate.io/) for vector search capabilities
- [Sentence Transformers](https://www.sbert.net/) for text embeddings
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for graph operations

## Contact

For questions and support, please open an issue in the GitHub repository.

---

Last updated: June 2025
=======
>>>>>>> main
