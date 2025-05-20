# src/data_ingestion/chunker.py
import uuid

DEFAULT_CHUNK_SIZE = 1000  # Characters
DEFAULT_CHUNK_OVERLAP = 100 # Characters
MIN_CHUNK_SIZE_THRESHOLD = 20 # Don't create chunks smaller than this unless it's the whole block

def chunk_content(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Splits a long text into smaller chunks with overlap.
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= text_len:
            break
        
        # Move start for the next chunk, considering overlap
        # We want the next chunk to start `chunk_overlap` characters before the end of the current one.
        # So, the step is `chunk_size - chunk_overlap`.
        start += (chunk_size - chunk_overlap)
        if start >= text_len: # Should not happen if end < text_len, but as a safeguard
            break
            
    return chunks

def chunk_conceptual_blocks(
    conceptual_blocks: list, 
    chunk_size: int = DEFAULT_CHUNK_SIZE, 
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list:
    """
    Takes a list of conceptual blocks and divides their content into
    manageable chunks.
    
    Each conceptual_block is a dict:
    {
        "source": str,
        "original_type": str ("latex" or "pdf"),
        "concept_type": str (e.g., "section", "theorem", "general_content"),
        "concept_name": str or None,
        "content": str (the text content of this conceptual block)
    }
    
    Output is a list of chunk dicts:
    {
        "chunk_id": str (unique ID for the chunk),
        "source": str,
        "original_type": str,
        "concept_type": str,
        "concept_name": str or None,
        "parent_block_content": str (original content of the conceptual block it came from)
        "chunk_text": str (the actual text of this chunk),
        "sequence_in_block": int (0-indexed sequence if a block was split)
    }
    """
    all_final_chunks = []

    for block in conceptual_blocks:
        content = block.get("content", "")
        if not content or not content.strip():
            continue # Skip empty conceptual blocks

        # For certain concept types, the "content" from the tagger might be the command itself
        # (e.g., section titles). These are usually short and might not need further chunking
        # or could be treated as metadata for subsequent general_content chunks.
        # For now, we'll chunk whatever content is provided.
        
        # If content is small enough, it's one chunk (unless it's trivially small)
        if len(content) <= chunk_size:
            if len(content.strip()) >= MIN_CHUNK_SIZE_THRESHOLD or len(content.strip()) > 0 : # Avoid empty or too small chunks
                all_final_chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "source": block["source"],
                    "original_type": block["original_type"],
                    "concept_type": block["concept_type"],
                    "concept_name": block.get("concept_name"),
                    "parent_block_content": content, # The whole block content is the parent
                    "chunk_text": content.strip(),
                    "sequence_in_block": 0
                })
        else:
            # Content is larger, so split it
            split_texts = chunk_content(content, chunk_size, chunk_overlap)
            for i, chunk_text in enumerate(split_texts):
                if chunk_text.strip(): # Ensure the split chunk is not just whitespace
                    all_final_chunks.append({
                        "chunk_id": str(uuid.uuid4()),
                        "source": block["source"],
                        "original_type": block["original_type"],
                        "concept_type": block["concept_type"], # Inherits concept type
                        "concept_name": block.get("concept_name"), # Inherits concept name
                        "parent_block_content": content, # Store original block content for context
                        "chunk_text": chunk_text.strip(),
                        "sequence_in_block": i
                    })
                    
    return all_final_chunks


if __name__ == '__main__':
    # Example conceptual blocks (simulating output from concept_tagger)
    sample_conceptual_blocks = [
        {
            "source": "doc1.tex", "original_type": "latex", 
            "concept_type": "section", "concept_name": "Introduction", 
            "content": "\\section{Introduction}" # Short, might become one chunk or be handled differently
        },
        {
            "source": "doc1.tex", "original_type": "latex",
            "concept_type": "general_content", "concept_name": None,
            "content": "This is the first paragraph of the introduction. It explains the main topic. " * 50 # Make it long
        },
        {
            "source": "doc1.tex", "original_type": "latex",
            "concept_type": "theorem", "concept_name": "Pythagorean Theorem",
            "content": "In a right-angled triangle, the square of the hypotenuse (the side opposite the right angle) " \
                       "is equal to the sum of the squares of the other two sides. $a^2 + b^2 = c^2$. " \
                       "This is a fundamental theorem in Euclidean geometry. " * 20 # Make it long
        },
        {
            "source": "doc2.pdf", "original_type": "pdf",
            "concept_type": "general_content", "concept_name": None,
            "content": "This is a short sentence from a PDF."
        },
        {
            "source": "doc3.tex", "original_type": "latex",
            "concept_type": "general_content", "concept_name": None,
            "content": "" # Empty content block
        }
    ]

    print("--- Chunking conceptual blocks ---")
    # Use smaller chunk size for easier inspection of the example
    final_chunks = chunk_conceptual_blocks(sample_conceptual_blocks, chunk_size=200, chunk_overlap=50)

    for i, chunk_data in enumerate(final_chunks):
        print(f"\n--- Chunk {i+1} (ID: {chunk_data['chunk_id']}) ---")
        print(f"  Source: {chunk_data['source']}")
        print(f"  Concept Type: {chunk_data['concept_type']}")
        print(f"  Concept Name: {chunk_data['concept_name']}")
        print(f"  Sequence in Block: {chunk_data['sequence_in_block']}")
        print(f"  Chunk Text (len {len(chunk_data['chunk_text'])}): '{chunk_data['chunk_text'][:100].replace(chr(10), ' ')}...'")
        # print(f"  Parent Block Content (first 100): '{chunk_data['parent_block_content'][:100].replace(chr(10), ' ')}...'")

    print(f"\nTotal final chunks created: {len(final_chunks)}")