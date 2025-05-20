# src/pipeline.py
import os
import asyncio
import json

from src.data_ingestion import document_loader
from src.data_ingestion import concept_tagger
from src.data_ingestion import chunker # This will use the langchain_text_splitters version
from src.data_ingestion import vector_store_manager
from src.retrieval import retriever
from src.generation import question_generator_rag
from src import config

# Constants from app.py, can be moved to config.py or passed as arguments
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_QUERY = "What is a vector space?"
DEFAULT_NUM_RETRIEVED_CHUNKS = 3
DEFAULT_NUM_QUESTIONS_TO_GENERATE = 2
DEFAULT_QUESTION_TYPE = "conceptual"
DEFAULT_SEARCH_TYPE = "hybrid"


async def run_ingestion_pipeline(processed_log_path: str):
    """
    Runs the data ingestion part of the pipeline (Phase 1).
    Processes new LaTeX documents and stores them in Weaviate.
    Returns the Weaviate client instance if successful, or None.
    """
    print("\n--- Phase 1: Data Ingestion & Storage (LaTeX Only) ---")
    print("\nStep 1.1: Loading and Parsing LaTeX Documents...")
    
    latex_documents_path = config.DATA_DIR_RAW_LATEX
    if not os.path.isdir(latex_documents_path):
        print(f"ERROR: LaTeX documents directory not found at {latex_documents_path}. Please create it and add .tex files.")
        return None

    try:
        print(f"Attempting to load only LaTeX documents from {latex_documents_path} (skipping already processed)...")
        all_parsed_docs = document_loader.load_and_parse_documents(
            process_pdfs=False,
            processed_docs_log_path=processed_log_path
        )
        parsed_docs_data = [doc for doc in all_parsed_docs if doc and doc.get("original_type") == "latex"]
    except Exception as e:
        print(f"ERROR: Failed during LaTeX document loading/parsing phase: {e}")
        import traceback
        traceback.print_exc()
        return None

    if not parsed_docs_data:
        print("No new LaTeX documents to process for ingestion.")
        # Try to get client for Phase 2 even if no new docs
        try:
            client = vector_store_manager.get_weaviate_client()
            print("Weaviate client obtained. No new documents were ingested in this run.")
            return client
        except Exception as e:
            print(f"Could not connect to Weaviate: {e}")
            return None
            
    print(f"Successfully parsed {len(parsed_docs_data)} new LaTeX documents.")

    print("\nStep 1.2: Concept/Topic Identification & Tagging...")
    all_conceptual_blocks = concept_tagger.tag_all_documents(parsed_docs_data)
    if not all_conceptual_blocks:
        if parsed_docs_data:
            print("ERROR: New LaTeX documents were parsed, but no conceptual blocks were identified. Exiting.")
            return None
        print("Warning: No conceptual blocks identified (likely no new parsed docs with content).")
    else:
        print(f"Identified {len(all_conceptual_blocks)} conceptual blocks from new LaTeX documents.")

    if not all_conceptual_blocks: # If still no blocks (e.g. parsed_docs_data was empty)
        try: # Try to get client for Phase 2
            client = vector_store_manager.get_weaviate_client()
            return client
        except Exception as e:
            print(f"Could not connect to Weaviate: {e}")
            return None


    print("\nStep 1.3: Text Chunking...")
    final_text_chunks = chunker.chunk_conceptual_blocks(
        all_conceptual_blocks,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )
    if not final_text_chunks:
        print("ERROR: No text chunks were created from the new LaTeX conceptual blocks. Exiting.")
        return None
    print(f"Created {len(final_text_chunks)} final text chunks from new LaTeX content.")

    print("\nStep 1.4 & 1.5: Initialize Weaviate, Embed and Store Chunks...")
    try:
        client = vector_store_manager.get_weaviate_client()
        vector_store_manager.embed_and_store_chunks(client, final_text_chunks)
        print(f"Data ingestion complete for new LaTeX content. Chunks stored in Weaviate class: {vector_store_manager.WEAVIATE_CLASS_NAME}")
        
        newly_ingested_filenames = list(set([doc['filename'] for doc in parsed_docs_data if 'filename' in doc]))
        document_loader.update_processed_docs_log(processed_log_path, newly_ingested_filenames)
        return client
    except ConnectionError as e:
        print(f"ERROR: Could not connect to Weaviate: {e}. Please ensure Weaviate is running.")
        return None
    except Exception as e:
        print(f"ERROR: Error during Weaviate initialization or data storage: {e}")
        import traceback
        traceback.print_exc()
        return None


async def run_retrieval_and_generation_pipeline(client):
    """
    Runs the retrieval and question generation part of the pipeline (Phase 2).
    Requires a Weaviate client instance.
    """
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
        for i, chunk_item in enumerate(retrieved_chunks):
            print(f"\n  --- Retrieved Chunk {i+1} ---")
            print(f"    ID: {chunk_item.get('_id', chunk_item.get('chunk_id', 'N/A'))}")
            print(f"    Source: {chunk_item.get('source_path', 'N/A')}")
            print(f"    Concept: {chunk_item.get('concept_name', 'N/A')}")
            print(f"    Text: \"{chunk_item.get('chunk_text', '')[:200]}...\"")
            for key, label in [('_certainty', 'Certainty'), ('_score', 'Score'), ('_distance', 'Distance')]:
                value = chunk_item.get(key)
                if value is not None:
                    try:
                        float_value = float(value)
                        print(f"    {label}: {float_value:.4f}")
                    except (ValueError, TypeError):
                        print(f"    {label}: {value} (raw)")
            if '_explainScore' in chunk_item and chunk_item['_explainScore'] is not None:
                print(f"    ExplainScore: {chunk_item['_explainScore']}")

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

async def run_full_pipeline():
    """
    Runs the complete RAG pipeline: ingestion, retrieval, and question generation.
    """
    print("Starting RAG System - Full Pipeline Execution...")
    
    processed_log_path = config.PROCESSED_DOCS_LOG_FILE
    
    # Run Ingestion Phase
    weaviate_client = await run_ingestion_pipeline(processed_log_path) # Ingestion is mostly sync, but keep async for consistency

    # Run Retrieval and Generation Phase
    if weaviate_client:
        await run_retrieval_and_generation_pipeline(weaviate_client)
    else:
        print("Ingestion phase failed or Weaviate client not available. Skipping retrieval and generation.")

    print("\n\n--- RAG System Pipeline Finished ---")
    await asyncio.sleep(0.25) # For resource cleanup

if __name__ == '__main__':
    # This allows running the pipeline directly for testing
    # Ensure .env is loaded if running this file directly
    if os.path.exists(os.path.join(os.path.dirname(__file__), '..', '.env')):
        from dotenv import load_dotenv
        print("pipeline.py: Found .env file in parent directory, loading.")
        load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
        # Re-evaluate config if .env might have changed it
        # This is a bit tricky; ideally config module handles this transparently.
        # For simplicity, assume config module is already loaded with .env values if app.py ran it.

    asyncio.run(run_full_pipeline())
