# src/app.py
import os # For checking if .env exists for API keys
from src.data_ingestion import document_loader
from src.data_ingestion import concept_tagger 
from src.data_ingestion import chunker 
from src.data_ingestion import vector_store_manager # New import
from src import config 

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

def main():
    print("Starting RAG System - Implementing Vector Storage...")

    # --- Step 1.1: Load and Parse Documents ---
    print("\nStep 1.1: Loading and Parsing Documents...")
    # Determine pdf_tool based on whether Mathpix keys are set, as an example
    pdf_tool_for_math_heavy_docs = "general"
    if config.MATHPIX_APP_ID and config.MATHPIX_APP_KEY:
        print("Mathpix API keys found in config, will attempt to use for math-heavy PDFs.")
        pdf_tool_for_math_heavy_docs = "mathpix"
    else:
        print("Mathpix API keys not found. Using general PDF parser for all PDFs.")
        
    parsed_docs_data = document_loader.load_and_parse_documents(pdf_math_tool=pdf_tool_for_math_heavy_docs)
    
    if not parsed_docs_data:
        print("No documents were processed by loader. Exiting.")
        return
    print(f"Successfully parsed {len(parsed_docs_data)} documents by loader.")

    # --- Step 1.2: Concept/Topic Identification & Tagging ---
    print("\nStep 1.2: Concept/Topic Identification & Tagging...")
    all_conceptual_blocks = concept_tagger.tag_all_documents(parsed_docs_data)
    
    if not all_conceptual_blocks:
        print("No conceptual blocks were identified.")
        # Decide if we should exit or proceed if some docs had content but no blocks were made
    else:
        print(f"Identified {len(all_conceptual_blocks)} conceptual blocks in total.")

    # --- Step 1.3: Text Chunking ---
    print("\nStep 1.3: Text Chunking...")
    final_text_chunks = []
    if all_conceptual_blocks:
        final_text_chunks = chunker.chunk_conceptual_blocks(
            all_conceptual_blocks,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        print(f"Created {len(final_text_chunks)} final text chunks.")
    else:
        print("No conceptual blocks to chunk.")
    
    if not final_text_chunks:
        print("No text chunks available for embedding. Exiting.")
        return

    # --- Step 1.4 & 1.5: Initialize Weaviate, Embed and Store Chunks ---
    print("\nStep 1.4 & 1.5: Initialize Weaviate, Embed and Store Chunks...")
    try:
        client = vector_store_manager.get_weaviate_client()
        # Schema creation is now handled within embed_and_store_chunks if needed
        # vector_store_manager.create_weaviate_schema(client) # Or call explicitly if preferred
        
        vector_store_manager.embed_and_store_chunks(client, final_text_chunks)
        
        print("\nEmbedding and storage process initiated.")
        print(f"Data should now be in Weaviate class: {vector_store_manager.WEAVIATE_CLASS_NAME}")

    except ConnectionError as e:
        print(f"Could not connect to Weaviate: {e}")
        print("Please ensure Weaviate is running (e.g., via 'docker-compose up -d').")
    except Exception as e:
        print(f"An error occurred during Weaviate initialization or data storage: {e}")
        import traceback
        traceback.print_exc()


    # --- TODO: Querying (Example) ---
    # print("\n--- Example Query (if data was added) ---")
    # if client:
    #     try:
    #         query_text = "vectors" # Example query
    #         query_embedding = vector_store_manager.generate_standard_embedding(query_text)
    #         if query_embedding:
    #             response = (
    #                 client.query
    #                 .get(vector_store_manager.WEAVIATE_CLASS_NAME, ["chunk_text", "source_path", "concept_name", "_additional {certainty distance id}"])
    #                 .with_near_vector({"vector": query_embedding})
    #                 .with_limit(3)
    #                 .do()
    #             )
    #             print("Vector search results for 'vectors':")
    #             import json
    #             print(json.dumps(response, indent=2))
    #     except Exception as e:
    #         print(f"Error during example query: {e}")


    print("\nPhase 1.4 & 1.5 (Embedding & Storage) foundational work complete.")

if __name__ == "__main__":
    main()