# src/data_ingestion/chunker.py
import uuid
from typing import List, Dict, Optional

# A simple text splitter for demonstration. 
# In a real scenario, you might use something more sophisticated like
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# or a custom regex-based splitter.

def simple_text_splitter(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    A basic text splitter.
    """
    if not text or chunk_size <= 0:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position for the next chunk
        # If overlap is too large, it might lead to issues, ensure step is positive
        step = chunk_size - chunk_overlap
        if step <= 0: # Avoid infinite loop if overlap >= chunk_size
            print(f"Warning: Chunk step is not positive (chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}). Advancing by chunk_size/2 or 1.")
            step = max(1, chunk_size // 2) 
        start += step
        
    return chunks

def chunk_conceptual_blocks(
    conceptual_blocks: List[Dict], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 150,
    min_chunk_size: int = 50 # Minimum size for a chunk to be considered valid
) -> List[Dict]:
    """
    Chunks the content of conceptual blocks into smaller text chunks.

    Args:
        conceptual_blocks: A list of dictionaries, where each dictionary represents a conceptual block
                           (e.g., from concept_tagger.py). Expected to have 'block_content',
                           'doc_id', 'source', 'original_type', 'concept_type', 'concept_name'.
        chunk_size: The target size for each chunk.
        chunk_overlap: The overlap between consecutive chunks.
        min_chunk_size: Minimum character length for a chunk to be kept.

    Returns:
        A list of dictionaries, where each dictionary represents a final text chunk
        ready for embedding.
    """
    final_text_chunks = []
    if not conceptual_blocks:
        print("Chunker: No conceptual blocks provided to chunk.")
        return final_text_chunks

    print(f"Chunker: Received {len(conceptual_blocks)} conceptual blocks to process.")

    for i, block in enumerate(conceptual_blocks):
        block_content = block.get("block_content", "")
        doc_id = block.get("doc_id", "unknown_doc")
        source = block.get("source", "unknown_source")
        original_type = block.get("original_type", "unknown_type")
        concept_type = block.get("concept_type", "unknown_concept")
        concept_name = block.get("concept_name", "Unnamed Concept")
        parent_block_id = block.get("block_id", str(uuid.uuid4())) # Use block_id as parent_block_id

        print(f"\nChunker: Processing conceptual block {i+1}/{len(conceptual_blocks)}:")
        print(f"  Doc ID: {doc_id}, Source: {source}")
        print(f"  Concept Type: {concept_type}, Concept Name: {concept_name}")
        print(f"  Parent Block ID: {parent_block_id}")
        print(f"  Block Content (first 100 chars): '{block_content[:100]}...'")
        print(f"  Block Content Length: {len(block_content)}")

        if not block_content or not block_content.strip():
            print("  Skipping this block due to empty or whitespace-only content.")
            continue

        # Using the simple_text_splitter. Replace with your actual splitter if different.
        # If using RecursiveCharacterTextSplitter:
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # sub_chunks_text = text_splitter.split_text(block_content)
        
        sub_chunks_text = simple_text_splitter(block_content, chunk_size, chunk_overlap)
        
        print(f"  Generated {len(sub_chunks_text)} sub-chunks from this block before filtering.")

        for j, chunk_text in enumerate(sub_chunks_text):
            chunk_text_stripped = chunk_text.strip()
            if len(chunk_text_stripped) >= min_chunk_size:
                chunk_id = str(uuid.uuid4())
                final_text_chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "source": source, # Keep original source for traceability
                    "original_type": original_type,
                    "concept_type": concept_type, # Concept type of the parent block
                    "concept_name": concept_name, # Concept name of the parent block
                    "parent_block_id": parent_block_id, # ID of the conceptual block this chunk came from
                    "chunk_text": chunk_text_stripped,
                    "sequence_in_block": j, # Order of this chunk within the conceptual block
                    "parent_block_content": block_content # Store the full parent block content for context
                })
                if j < 2: # Print first few chunks for brevity
                    print(f"    Added chunk {j+1}: '{chunk_text_stripped[:80]}...' (Length: {len(chunk_text_stripped)})")
            else:
                print(f"    Skipped sub-chunk {j+1} due to insufficient length after stripping (Length: {len(chunk_text_stripped)}, Min: {min_chunk_size}). Original: '{chunk_text[:80]}...'")
        
        if not sub_chunks_text and block_content.strip():
            print(f"  WARNING: No sub-chunks generated from non-empty block content. Check splitter logic or content nature.")


    print(f"\nChunker: Finished processing. Total final text chunks created: {len(final_text_chunks)}")
    return final_text_chunks

if __name__ == '__main__':
    # Example Usage
    sample_conceptual_blocks = [
        {
            "doc_id": "doc1", "source": "source1.tex", "original_type": "latex",
            "block_id": "block1-1", "concept_type": "section", "concept_name": "Introduction",
            "block_content": "This is the first section. It has enough text to be split into multiple chunks hopefully. Let's add more sentences to make sure it exceeds the chunk size. This sentence makes it longer. And another one for good measure."
        },
        {
            "doc_id": "doc1", "source": "source1.tex", "original_type": "latex",
            "block_id": "block1-2", "concept_type": "subsection", "concept_name": "Background",
            "block_content": "This is a subsection with shorter content."
        },
        {
            "doc_id": "doc2", "source": "source2.pdf", "original_type": "pdf",
            "block_id": "block2-1", "concept_type": "full_document", "concept_name": "Full PDF content",
            "block_content": "A very short block, possibly less than min_chunk_size." # Test min_chunk_size
        },
        {
            "doc_id": "doc3", "source": "source3.txt", "original_type": "text",
            "block_id": "block3-1", "concept_type": "paragraph", "concept_name": "First Paragraph",
            "block_content": "    " # Whitespace only
        }
    ]

    print("--- Testing Chunker ---")
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
            print(f"  Chunk ID: {chunk_data['chunk_id']}")
            print(f"  Doc ID: {chunk_data['doc_id']}")
            print(f"  Source: {chunk_data['source']}")
            print(f"  Concept: {chunk_data['concept_name']} ({chunk_data['concept_type']})")
            print(f"  Parent Block ID: {chunk_data['parent_block_id']}")
            print(f"  Sequence in Block: {chunk_data['sequence_in_block']}")
            print(f"  Text: \"{chunk_data['chunk_text']}\"")
            # print(f"  Parent Block Content: \"{chunk_data['parent_block_content'][:50]}...\"")

    else:
        print("No final chunks were generated by the chunker in the demo.")
