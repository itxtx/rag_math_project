# src/data_ingestion/chunker.py
import uuid
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter # New import
import os

def chunk_conceptual_blocks(
    conceptual_blocks: List[Dict], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 150,
    min_chunk_size: int = 50, # Minimum size for a chunk to be considered valid
    separators: Optional[List[str]] = None # For RecursiveCharacterTextSplitter
) -> List[Dict]:
    """
    Chunks the content of conceptual blocks into smaller text chunks using RecursiveCharacterTextSplitter.

    Args:
        conceptual_blocks: A list of dictionaries, where each dictionary represents a conceptual block.
        chunk_size: The target size for each chunk (passed to splitter).
        chunk_overlap: The overlap between consecutive chunks (passed to splitter).
        min_chunk_size: Minimum character length for a chunk to be kept after splitting.
        separators: Optional list of separators for RecursiveCharacterTextSplitter. 
                    Defaults to ["\n\n", "\n", " ", ""].

    Returns:
        A list of dictionaries, where each dictionary represents a final text chunk.
    """
    final_text_chunks = []
    if not conceptual_blocks:
        print("Chunker: No conceptual blocks provided to chunk.")
        return final_text_chunks

    print(f"Chunker: Received {len(conceptual_blocks)} conceptual blocks to process.")

    # Initialize the text splitter
    if separators is None:
        separators = ["\n\n", "\n", "\r", "\t", " ", "(", ")", "{", "}", "[", "]", ".", ",", ";", ":", "-", "?", "!"] # More comprehensive default
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False, # Treat separators as literals
        separators=separators
    )

    for i, block in enumerate(conceptual_blocks):
        block_content = block.get("block_content", "")
        doc_id = block.get("doc_id", "unknown_doc")
        source = block.get("source", "unknown_source")
        original_type = block.get("original_type", "unknown_type")
        concept_type = block.get("concept_type", "unknown_concept")
        concept_name = block.get("concept_name", "Unnamed Concept")
        parent_block_id = block.get("block_id", str(uuid.uuid4()))

        print(f"\nChunker: Processing conceptual block {i+1}/{len(conceptual_blocks)}:")
        print(f"  Doc ID: {doc_id}, Source: {source}")
        print(f"  Concept Type: {concept_type}, Concept Name: {concept_name}")
        print(f"  Parent Block ID: {parent_block_id}")
        print(f"  Block Content Length: {len(block_content)}")
        # print(f"  Block Content (first 100 chars): '{block_content[:100]}...'") # Keep this for less verbose output

        if not block_content or not block_content.strip():
            print("  Skipping this block due to empty or whitespace-only content.")
            continue
        
        sub_chunks_text = text_splitter.split_text(block_content)
        
        print(f"  Generated {len(sub_chunks_text)} sub-chunks from this block before filtering.")

        for j, chunk_text in enumerate(sub_chunks_text):
            chunk_text_stripped = chunk_text.strip() # Stripping is important
            if len(chunk_text_stripped) >= min_chunk_size:
                chunk_id = str(uuid.uuid4())
                final_text_chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "source": source,
                    "filename": os.path.basename(source), # Add filename for easier logging if needed
                    "original_type": original_type,
                    "concept_type": concept_type,
                    "concept_name": concept_name,
                    "parent_block_id": parent_block_id,
                    "chunk_text": chunk_text_stripped,
                    "sequence_in_block": j,
                    "parent_block_content": block_content 
                })
                if j < 1: # Print first chunk for brevity
                    print(f"    Added chunk {j+1}: '{chunk_text_stripped[:80]}...' (Length: {len(chunk_text_stripped)})")
            else:
                print(f"    Skipped sub-chunk {j+1} due to insufficient length after stripping (Length: {len(chunk_text_stripped)}, Min: {min_chunk_size}). Original: '{chunk_text[:80]}...'")
        
        if not sub_chunks_text and block_content.strip(): # If splitter returned nothing for non-empty content
            print(f"  WARNING: RecursiveCharacterTextSplitter returned no chunks for non-empty block. Content: '{block_content[:100]}...'")


    print(f"\nChunker: Finished processing. Total final text chunks created: {len(final_text_chunks)}")
    return final_text_chunks

if __name__ == '__main__':
    sample_conceptual_blocks = [
        {
            "doc_id": "doc1", "source": "source1.tex", "original_type": "latex",
            "block_id": "block1-1", "concept_type": "section", "concept_name": "Introduction",
            "block_content": "This is the first section. It has enough text to be split into multiple chunks hopefully. Let's add more sentences to make sure it exceeds the chunk size. This sentence makes it longer. And another one for good measure. This should definitely be more than one chunk.\n\nA new paragraph here. Another sentence. And one more."
        },
        {
            "doc_id": "doc1", "source": "source1.tex", "original_type": "latex",
            "block_id": "block1-2", "concept_type": "subsection", "concept_name": "Background",
            "block_content": "This is a subsection with shorter content, but still long enough for one chunk."
        },
        {
            "doc_id": "doc2", "source": "source2.pdf", "original_type": "pdf",
            "block_id": "block2-1", "concept_type": "full_document", "concept_name": "Full PDF content",
            "block_content": "Short." # Test min_chunk_size
        }
    ]

    print("--- Testing Chunker with RecursiveCharacterTextSplitter ---")
    final_chunks = chunk_conceptual_blocks(
        sample_conceptual_blocks, 
        chunk_size=100, 
        chunk_overlap=20,
        min_chunk_size=10
    )

    if final_chunks:
        print(f"\n--- Generated Chunks ({len(final_chunks)}) ---")
        for i, chunk_data in enumerate(final_chunks):
            print(f"\nChunk {i+1}:")
            print(f"  Text: \"{chunk_data['chunk_text']}\"")
    else:
        print("No final chunks were generated by the chunker in the demo.")


