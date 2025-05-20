# src/app.py
import os
import asyncio # For running async functions
import json # For pretty printing results

from src.data_ingestion import document_loader
from src.data_ingestion import concept_tagger
from src.data_ingestion import chunker
from src.data_ingestion import vector_store_manager
from src.retrieval import retriever
from src.generation import question_generator_rag
from src import config

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

# --- Configuration for Phase 2 ---
DEFAULT_QUERY = "What is a vector space?" # Adjusted for potential linear_algebra.tex content
DEFAULT_NUM_RETRIEVED_CHUNKS = 3
DEFAULT_NUM_QUESTIONS_TO_GENERATE = 2
DEFAULT_QUESTION_TYPE = "conceptual"
DEFAULT_SEARCH_TYPE = "hybrid"


async def main():
    print("Starting RAG System - LaTeX Only Ingestion Pipeline Demo...")
    client = None

    # --- Phase 1: Data Ingestion and Storage (LaTeX Only) ---
    print("\n--- Phase 1: Data Ingestion & Storage (LaTeX Only) ---")

    # Step 1.1: Load and Parse LaTeX Documents
    print("\nStep 1.1: Loading and Parsing LaTeX Documents...")
    # Modify to only process LaTeX. We can pass specific flags or adjust document_loader
    # For simplicity here, we'll assume document_loader can be told to skip PDFs
    # or we'll filter its output.
    # A more direct way is to call a LaTeX-specific function if available,
    # or ensure DATA_DIR_RAW_PDFS is ignored by document_loader.
    # Let's assume load_and_parse_documents can take flags to specify types.
    # If not, we'd call a more specific function like `process_latex_documents`.

    # For this modification, we'll directly call the LaTeX processing part of document_loader
    # This assumes `process_latex_documents` exists and returns a similar structure.
    # If `document_loader.load_and_parse_documents` is monolithic, we'd filter its results.
    # Let's assume a more direct call for clarity of intent:
    
    latex_documents_path = getattr(config, 'DATA_DIR_RAW_LATEX', 'data/raw_latex')
    if not os.path.isdir(latex_documents_path):
        print(f"ERROR: LaTeX documents directory not found at {latex_documents_path}. Please create it and add .tex files.")
        return

    # Assuming process_latex_documents returns a list of parsed doc data
    # This might require a small refactor in document_loader if it's not already modular
    try:
        # This is a conceptual change. If your document_loader.process_latex_documents
        # isn't structured to be called directly like this and return the expected list of dicts,
        # you'd need to adjust document_loader.py or how you call it.
        # For now, let's simulate this specific call.
        # parsed_docs_data = document_loader.process_latex_documents() # Idealized call
        
        # More realistically, if load_and_parse_documents is the entry point:
        # We'll tell it to only process latex by effectively nullifying pdf_processing_tool
        # and relying on its internal logic to skip PDF if no tool or path is valid.
        # Or, better, add a flag to load_and_parse_documents.
        # For now, let's filter the output if it processes both.
        print(f"Attempting to load only LaTeX documents from {latex_documents_path}...")
        all_parsed_docs = document_loader.load_and_parse_documents(process_pdfs=False) # Assuming such a flag exists or can be added

        parsed_docs_data = [doc for doc in all_parsed_docs if doc.get("original_type") == "latex"]

    except Exception as e:
        print(f"ERROR: Failed during LaTeX document loading/parsing phase: {e}")
        import traceback
        traceback.print_exc()
        return

    if not parsed_docs_data:
        print("ERROR: No LaTeX documents were successfully parsed or found. Exiting.")
        return
    print(f"Successfully parsed {len(parsed_docs_data)} LaTeX documents.")

    # Step 1.2: Concept/Topic Identification & Tagging for LaTeX content
    print("\nStep 1.2: Concept/Topic Identification & Tagging (for LaTeX content)...")
    all_conceptual_blocks = concept_tagger.tag_all_documents(parsed_docs_data) # This will now only process LaTeX data

    if not all_conceptual_blocks:
        print("Warning: No conceptual blocks were identified from the parsed LaTeX documents.")
        # If this is critical, we can exit. For now, we'll let chunking handle empty input.
        # However, if parsing was successful but no blocks, it's an issue.
        # Let's assume if parsed_docs_data was non-empty, we expect *some* blocks or it's an issue.
        if parsed_docs_data: # If we had parsed docs but got no blocks
             print("ERROR: LaTeX documents were parsed, but no conceptual blocks were identified. This might indicate an issue in the tagging process for LaTeX content. Exiting.")
             return
        # If parsed_docs_data was also empty, the earlier check would have exited.
    else:
        print(f"Identified {len(all_conceptual_blocks)} conceptual blocks from LaTeX documents.")

    # Step 1.3: Text Chunking for LaTeX-derived blocks
    print("\nStep 1.3: Text Chunking (for LaTeX-derived blocks)...")
    final_text_chunks = chunker.chunk_conceptual_blocks(
        all_conceptual_blocks, # This list now only contains LaTeX-derived blocks
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )

    if not final_text_chunks:
        print("ERROR: No text chunks were created from the LaTeX conceptual blocks. This could be due to empty blocks or a chunking issue. Exiting.")
        return
    print(f"Created {len(final_text_chunks)} final text chunks from LaTeX content.")

    # Step 1.4 & 1.5: Initialize Weaviate, Embed and Store Chunks
    print("\nStep 1.4 & 1.5: Initialize Weaviate, Embed and Store Chunks (from LaTeX content)...")
    try:
        client = vector_store_manager.get_weaviate_client()
        vector_store_manager.embed_and_store_chunks(client, final_text_chunks)
        print(f"Data ingestion complete for LaTeX content. Chunks stored in Weaviate class: {vector_store_manager.WEAVIATE_CLASS_NAME}")
    except ConnectionError as e:
        print(f"ERROR: Could not connect to Weaviate: {e}. Please ensure Weaviate is running.")
        return
    except Exception as e:
        print(f"ERROR: Error during Weaviate initialization or data storage for LaTeX content: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Phase 2: Retrieval and Question Generation ---
    print("\n\n--- Phase 2: Retrieval & Question Generation (based on LaTeX content) ---")
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
        # This is not necessarily an error for the app to exit, but Phase 2.4 won't run.
    else:
        print(f"Successfully retrieved {len(retrieved_chunks)} chunks:")
        for i, chunk in enumerate(retrieved_chunks):
            print(f"\n  --- Retrieved Chunk {i+1} ---")
            print(f"    ID: {chunk.get('_id', chunk.get('chunk_id', 'N/A'))}")
            print(f"    Source: {chunk.get('source_path', 'N/A')}")
            print(f"    Concept: {chunk.get('concept_name', 'N/A')}")
            print(f"    Text: \"{chunk.get('chunk_text', '')[:200]}...\"")

            for key, label in [('_certainty', 'Certainty'), ('_score', 'Score'), ('_distance', 'Distance')]:
                value = chunk.get(key)
                if value is not None:
                    try:
                        float_value = float(value)
                        print(f"    {label}: {float_value:.4f}")
                    except (ValueError, TypeError):
                        print(f"    {label}: {value} (raw)")
            
            if '_explainScore' in chunk and chunk['_explainScore'] is not None:
                print(f"    ExplainScore: {chunk['_explainScore']}")

        print(f"\nStep 2.3: Initializing RAGQuestionGenerator...")
        try:
            question_gen = question_generator_rag.RAGQuestionGenerator()
            print("RAGQuestionGenerator initialized.")
        except Exception as e:
            print(f"Error initializing RAGQuestionGenerator: {e}")
            return # Exit if generator can't be initialized

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
            print("No questions were generated from the retrieved context (LLM might have failed or returned empty).")

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
