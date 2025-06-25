# src/data_ingestion/chunker.py
import uuid
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def chunk_conceptual_blocks(
    conceptual_blocks: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    min_chunk_size: int = 50,
    separators: Optional[List[str]] = None
) -> List[Dict]:
    """
    Chunks the content of conceptual blocks into smaller text chunks using RecursiveCharacterTextSplitter.
    """
    final_text_chunks = []
    if not conceptual_blocks:
        print("Chunker: No conceptual blocks provided to chunk.")
        return final_text_chunks

    print(f"Chunker: Received {len(conceptual_blocks)} conceptual blocks to process.")

    if separators is None:
        separators = ["\n\n", "\n", "\r", "\t", " ", "(", ")", "{", "}", "[", "]", ".", ",", ";", ":", "-", "?", "!"]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=separators
    )

    print(f"DEBUG: Created {len(conceptual_blocks)} blocks total")
    for i, block in enumerate(conceptual_blocks):
        block_content = block.get("block_content", "")

        # This print block is for debugging the chunker's input
        print(f"DEBUG: Processing node {block.get('block_id', 'N/A')}:")
        print(f"  Type: {block.get('concept_type')}")
        print(f"  Text length: {len(block_content)}")
        if len(block_content) > 0:
            preview_text = block_content[:120].replace('\n', ' ')
            print(f"  Text preview: {preview_text}...")

        if not block_content or not block_content.strip():
            print(f"  Skipping this block due to empty or whitespace-only content.")
            continue

        print(f"  Created block with text length: {len(block_content)}")
        sub_chunks_text = text_splitter.split_text(block_content)

        for j, chunk_text in enumerate(sub_chunks_text):
            chunk_text_stripped = chunk_text.strip()
            if len(chunk_text_stripped) >= min_chunk_size:
                final_text_chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": block.get("doc_id", "unknown_doc"),
                    "source": block.get("source", "unknown_source"),
                    "filename": os.path.basename(block.get("source", "unknown_source")),
                    "parent_block_id": block.get("block_id"),
                    "concept_type": block.get("concept_type"),
                    "concept_name": block.get("concept_name"),
                    "chunk_text": chunk_text_stripped,
                    "parent_block_content": block_content,
                })

    print(f"\nChunker: Finished processing. Total final text chunks created: {len(final_text_chunks)}")
    return final_text_chunks