# src/app.py
from src.data_ingestion import document_loader
from src.data_ingestion import concept_tagger
from src.data_ingestion import chunker # New import
from src import config 

DEFAULT_CHUNK_SIZE = 1000  # Define or import from config if preferred
DEFAULT_CHUNK_OVERLAP = 150

def main():
    print("Starting RAG System - Implementing Text Chunking...")

    # --- Step 1.1: Load and Parse Documents ---
    print("\nStep 1.1: Loading and Parsing Documents...")
    pdf_tool_for_math_heavy_docs = "general" 
    parsed_docs_data = document_loader.load_and_parse_documents(pdf_math_tool=pdf_tool_for_math_heavy_docs)
    
    if not parsed_docs_data:
        print("No documents were processed by loader.")
        return
    print(f"Successfully parsed {len(parsed_docs_data)} documents by loader.")

    # --- Step 1.2: Concept/Topic Identification & Tagging ---
    print("\nStep 1.2: Concept/Topic Identification & Tagging...")
    all_conceptual_blocks = concept_tagger.tag_all_documents(parsed_docs_data)
    
    if not all_conceptual_blocks:
        print("No conceptual blocks were identified.")
    else:
        print(f"Identified {len(all_conceptual_blocks)} conceptual blocks in total.")

    # --- Step 1.3: Text Chunking ---
    print("\nStep 1.3: Text Chunking...")
    if all_conceptual_blocks:
        final_text_chunks = chunker.chunk_conceptual_blocks(
            all_conceptual_blocks,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        print(f"Created {len(final_text_chunks)} final text chunks.")
        # For demonstration, print info about a few chunks
        for i, chunk_data in enumerate(final_text_chunks[:3]): # Print first 3 chunks
            print(f"  Chunk {i+1} (ID: {chunk_data['chunk_id']}):")
            print(f"    Source: {chunk_data['source']}")
            print(f"    Concept: {chunk_data['concept_type']} (Name: {chunk_data.get('concept_name', 'N/A')})")
            print(f"    Seq: {chunk_data['sequence_in_block']}")
            print(f"    Text (len {len(chunk_data['chunk_text'])}): '{chunk_data['chunk_text'][:70].replace(chr(10), ' ')}...'")
        if len(final_text_chunks) > 3:
            print(f"  ... and {len(final_text_chunks) - 3} more chunks.")
    else:
        print("No conceptual blocks to chunk.")
        final_text_chunks = []


    # --- Next Steps (Placeholders) ---
    print("\n--- TODO: Implement Next Steps ---")
    
    # Step 1.4: Initialize Weaviate Client & Schema
    # client = vector_store_manager.get_weaviate_client()
    # vector_store_manager.ensure_schema_exists(client) # Define schema for these chunks
    print("Step 1.4: Initialize Weaviate Client & Schema (Not Implemented Yet)")

    # Step 1.5: Embedding Generation & Vector Storage (will take final_text_chunks)
    # vector_store_manager.embed_and_store_chunks(final_text_chunks, client)
    print("Step 1.5: Embedding Generation & Vector Storage (Not Implemented Yet)")
    
    print("\nPhase 1.3 (Text Chunking) foundational work complete.")

if __name__ == "__main__":
    main()