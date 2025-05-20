# src/app.py
import os
import asyncio # For running async functions
import json # For pretty printing results

from src.data_ingestion import document_loader
from src.data_ingestion import concept_tagger
from src.data_ingestion import chunker
from src.data_ingestion import vector_store_manager
from src.retrieval import retriever # New import
from src.generation import question_generator_rag # New import
from src import config

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

# --- Configuration for Phase 2 ---
DEFAULT_QUERY = "What is the first law of thermodynamics and its implications?"
DEFAULT_NUM_RETRIEVED_CHUNKS = 3
DEFAULT_NUM_QUESTIONS_TO_GENERATE = 2
DEFAULT_QUESTION_TYPE = "conceptual"
DEFAULT_SEARCH_TYPE = "hybrid"


async def main():
    print("Starting RAG System - Full Pipeline Demo...")
    client = None

    # --- Phase 1: Data Ingestion and Storage ---
    print("\n--- Phase 1: Data Ingestion & Storage ---")
    print("\nStep 1.1: Loading and Parsing Documents...")
    pdf_tool_for_math_heavy_docs = "general"
    if config.MATHPIX_APP_ID and config.MATHPIX_APP_KEY:
        print("Mathpix API keys found. Using Mathpix for math-heavy PDFs.")
        pdf_tool_for_math_heavy_docs = "mathpix"
    else:
        print("Mathpix API keys not found. Using general PDF parser.")

    parsed_docs_data = document_loader.load_and_parse_documents(pdf_math_tool=pdf_tool_for_math_heavy_docs)
    if not parsed_docs_data:
        print("No documents processed. Exiting.")
        return
    print(f"Successfully parsed {len(parsed_docs_data)} documents.")

    print("\nStep 1.2: Concept/Topic Identification & Tagging...")
    all_conceptual_blocks = concept_tagger.tag_all_documents(parsed_docs_data)
    if not all_conceptual_blocks:
        print("No conceptual blocks identified. Using raw document content as fallback.")
        all_conceptual_blocks = []
        for doc_data in parsed_docs_data:
            if doc_data.get("parsed_content"):
                 all_conceptual_blocks.append({
                    "doc_id": doc_data["doc_id"],
                    "source": doc_data["source"],
                    "original_type": doc_data["original_type"],
                    "concept_type": "full_document_content",
                    "concept_name": os.path.basename(doc_data["source"]),
                    "block_content": doc_data["parsed_content"],
                    "block_order": 0
                })
        if not all_conceptual_blocks:
            print("No content to process further. Exiting.")
            return
        print(f"Using raw document content for {len(all_conceptual_blocks)} documents as fallback.")
    else:
        print(f"Identified {len(all_conceptual_blocks)} conceptual blocks.")

    print("\nStep 1.3: Text Chunking...")
    final_text_chunks = chunker.chunk_conceptual_blocks(
        all_conceptual_blocks,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )
    if not final_text_chunks:
        print("No text chunks created. Exiting.")
        return
    print(f"Created {len(final_text_chunks)} final text chunks.")

    print("\nStep 1.4 & 1.5: Initialize Weaviate, Embed and Store Chunks...")
    try:
        client = vector_store_manager.get_weaviate_client()
        vector_store_manager.embed_and_store_chunks(client, final_text_chunks)
        print(f"Data ingestion complete. Chunks stored in Weaviate class: {vector_store_manager.WEAVIATE_CLASS_NAME}")
    except ConnectionError as e:
        print(f"Could not connect to Weaviate: {e}. Please ensure Weaviate is running.")
        return
    except Exception as e:
        print(f"Error during Weaviate initialization or data storage: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Phase 2: Retrieval and Question Generation ---
    print("\n\n--- Phase 2: Retrieval & Question Generation ---")
    if not client:
        print("Weaviate client not available. Cannot proceed with Phase 2.")
        return

    print(f"\nStep 2.1: Initializing Retriever for class '{vector_store_manager.WEAVIATE_CLASS_NAME}'...")
    try:
        doc_retriever = retriever.Retriever(
            weaviate_client=client,
            weaviate_class_name=vector_store_manager.WEAVIATE_CLASS_NAME
        )
        print("Retriever initialized.")
    except Exception as e:
        print(f"Error initializing Retriever: {e}")
        return

    print(f"\nStep 2.2: Retrieving chunks for query: '{DEFAULT_QUERY}' using '{DEFAULT_SEARCH_TYPE}' search...")
    retrieved_chunks = doc_retriever.search(
        query_text=DEFAULT_QUERY,
        search_type=DEFAULT_SEARCH_TYPE,
        limit=DEFAULT_NUM_RETRIEVED_CHUNKS,
        additional_properties=["id", "distance", "certainty", "score", "explainScore"]
    )

    if not retrieved_chunks:
        print("No relevant chunks retrieved for the query. Cannot generate questions.")
    else:
        print(f"Successfully retrieved {len(retrieved_chunks)} chunks:")
        for i, chunk in enumerate(retrieved_chunks):
            print(f"\n  --- Retrieved Chunk {i+1} ---")
            print(f"    ID: {chunk.get('_id', chunk.get('chunk_id', 'N/A'))}")
            print(f"    Source: {chunk.get('source_path', 'N/A')}")
            print(f"    Concept: {chunk.get('concept_name', 'N/A')}")
            print(f"    Text: \"{chunk.get('chunk_text', '')[:200]}...\"")

            # --- FIXED VALUEERROR and TYPEERROR HERE ---
            for key, label in [('_certainty', 'Certainty'), ('_score', 'Score'), ('_distance', 'Distance')]:
                value = chunk.get(key)
                if value is not None:
                    try:
                        # Attempt to convert to float for formatting
                        float_value = float(value)
                        print(f"    {label}: {float_value:.4f}")
                    except (ValueError, TypeError):
                        # If conversion fails, print the raw value
                        print(f"    {label}: {value} (raw)")
            
            if '_explainScore' in chunk and chunk['_explainScore'] is not None:
                print(f"    ExplainScore: {chunk['_explainScore']}")
            # --- END OF FIX ---

        print(f"\nStep 2.3: Initializing RAGQuestionGenerator...")
        try:
            question_gen = question_generator_rag.RAGQuestionGenerator()
            print("RAGQuestionGenerator initialized.")
        except Exception as e:
            print(f"Error initializing RAGQuestionGenerator: {e}")
            return

        print(f"\nStep 2.4: Generating {DEFAULT_NUM_QUESTIONS_TO_GENERATE} '{DEFAULT_QUESTION_TYPE}' questions...")
        generated_questions = await question_gen.generate_questions(
            context_chunks=retrieved_chunks,
            num_questions=DEFAULT_NUM_QUESTIONS_TO_GENERATE,
            question_type=DEFAULT_QUESTION_TYPE
        )

        if generated_questions:
            print("\n--- Generated Questions ---")
            for i, q_text in enumerate(generated_questions):
                print(f"  Q{i+1}: {q_text}")
        else:
            print("No questions were generated from the retrieved context.")

    print("\n\n--- RAG System Demo Finished ---")

if __name__ == "__main__":
    if os.path.exists(".env"):
        print("Found .env file, attempting to load environment variables.")
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value
                    if not hasattr(config, key):
                         setattr(config, key, value)
        print("Environment variables from .env potentially loaded.")

    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred in the main application flow: {e}")
        import traceback
        traceback.print_exc()
