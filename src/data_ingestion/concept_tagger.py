# src/data_ingestion/concept_tagger.py
import re
from typing import List, Dict, Tuple, Optional
import uuid
import os
# Updated patterns to match the output of the current latex_parser.py
# The parser converts \section{Title} to "§ TITLE", etc.
LATEX_CONCEPT_PATTERNS = {
    "section": r"^§\s+(.*?)\s*$",
    "subsection": r"^§\.§\s+(.*?)\s*$",
    "subsubsection": r"^§\.§\.§\s+(.*?)\s*$",
    # The following patterns are for raw LaTeX and will NOT work with the current
    # latex_parser.py output, which strips these commands/environments.
    # They are commented out to prevent false negatives or errors.
    # If these concepts need to be tagged, either latex_parser.py must preserve
    # these structures, or a different tagging strategy is needed here.
    # "paragraph_title": r"\\paragraph\*?\{(.*?)\}", # Current latex_parser.py removes \\paragraph command
    # "environment": r"\\begin\{(theorem|definition|lemma|proof|example|remark|corollary|proposition|problem|solution)\}\s*(?:\[(.*?)\])?([\s\S]*?)\\end\{\1\}"
    # Current latex_parser.py removes \\begin{} and \\end{} tags for these environments, outputting only their content.
}

# If you need to tag content that was previously in theorem-like environments,
# and latex_parser.py now outputs their content directly, you would need a different
# strategy in this concept_tagger.py, perhaps based on keywords or formatting
# if any is preserved by latex_parser.py (e.g., if "Theorem:" precedes the content).
# This is beyond simple regex pattern replacement for LATEX_CONCEPT_PATTERNS.


def identify_conceptual_blocks_in_document(doc_id: str, source: str, original_type: str, parsed_content: str) -> List[Dict]:
    """
    Identifies conceptual blocks (like sections, theorems) in parsed document content
    based on predefined regex patterns.

    Args:
        doc_id: Unique identifier for the document.
        source: Original source path of the document.
        original_type: Type of the original document (e.g., "latex", "pdf").
        parsed_content: The string content of the parsed document.

    Returns:
        A list of dictionaries, where each dictionary represents a conceptual block.
    """
    conceptual_blocks = []
    current_pos = 0
    block_order = 0

    # Store all found matches with their start positions and types
    matches_with_positions = []

    patterns_to_use = {}
    if original_type == "latex":
        patterns_to_use = LATEX_CONCEPT_PATTERNS
    # Add elif for "pdf" or "text" if specific patterns are needed for them later
    else: # Default or other types
        # For now, if not LaTeX, we might not have specific patterns,
        # or we could define generic paragraph splitting as a fallback.
        # Let's assume for now only LaTeX has these rich concept patterns.
        pass

    if not patterns_to_use: # If no patterns for this doc type, treat whole doc as one block
        if parsed_content and parsed_content.strip():
            conceptual_blocks.append({
                "doc_id": doc_id,
                "source": source,
                "original_type": original_type,
                "block_id": str(uuid.uuid4()),
                "concept_type": "full_document",
                "concept_name": f"Full content of {os.path.basename(source)}",
                "block_content": parsed_content.strip(),
                "block_order": block_order
            })
        return conceptual_blocks


    for concept_key, pattern in patterns_to_use.items():
        try:
            for match in re.finditer(pattern, parsed_content, re.MULTILINE):
                # Group 1 is typically the title/name if pattern is (.*?)
                # For environment, group 1 is env name, group 2 is opt title, group 3 is content
                title = None
                content_start_offset = 0 # Where the actual content for this block begins relative to match.start()
                
                if concept_key in ["section", "subsection", "subsubsection"]:
                    title = match.group(1).strip() if match.group(1) else f"Unnamed {concept_key}"
                    content_start_offset = match.end() # Content starts after the matched title line
                elif concept_key == "paragraph_title": # This pattern is currently commented out
                    title = match.group(1).strip() if match.group(1) else "Unnamed Paragraph"
                    # Content for paragraph starts after the title line.
                    # This assumes paragraph content is not captured by the regex itself.
                    content_start_offset = match.end()
                elif concept_key == "environment": # This pattern is currently commented out
                    env_name = match.group(1)
                    opt_title = match.group(2)
                    env_content = match.group(3) # The regex captures the content directly
                    title = opt_title.strip() if opt_title else f"{env_name.capitalize()}"
                    # For environments where regex captures content, we use that.
                    # The match.start() is the beginning of \begin{...}
                    # We'll store the full matched block (including tags) if we want to preserve them.
                    # However, the current latex_parser strips these.
                    # This part needs rethinking if tags are stripped.
                    # For now, let's assume if this pattern was active, it would capture the raw block.
                    # matches_with_positions.append({
                    #     "start": match.start(),
                    #     "match_obj": match, # Store the match object
                    #     "concept_type": env_name, # e.g., "theorem", "definition"
                    #     "title": title,
                    #     "is_environment_block": True, # Flag for special handling
                    #     "env_content": env_content.strip() # Store pre-captured content
                    # })
                    # continue # Skip default content splitting for these
                    pass # This block is effectively disabled

                if title: # For section-like and paragraph_title
                    matches_with_positions.append({
                        "start": match.start(),
                        "match_obj": match,
                        "concept_type": concept_key,
                        "title": title,
                        "is_environment_block": False,
                        "content_start_offset": content_start_offset
                    })
        except Exception as e:
            print(f"Error during regex matching for {concept_key} with pattern '{pattern}': {e}")


    # Sort matches by their start position
    matches_with_positions.sort(key=lambda x: x["start"])

    # Iterate through sorted matches to define blocks
    # The content of a block is from its title's end to the start of the next title,
    # or to the end of the document.
    for i, current_match_info in enumerate(matches_with_positions):
        match_obj = current_match_info["match_obj"]
        concept_type = current_match_info["concept_type"]
        title = current_match_info["title"]
        
        block_start_pos = current_match_info["content_start_offset"]

        # Determine end of the current block's content
        if i + 1 < len(matches_with_positions):
            next_match_info = matches_with_positions[i+1]
            block_end_pos = next_match_info["start"] # Content ends where next block's title starts
        else:
            block_end_pos = len(parsed_content) # Last block goes to end of document

        block_content_text = parsed_content[block_start_pos:block_end_pos].strip()

        if block_content_text: # Only add block if it has actual content
            conceptual_blocks.append({
                "doc_id": doc_id,
                "source": source,
                "original_type": original_type,
                "block_id": str(uuid.uuid4()),
                "concept_type": concept_type,
                "concept_name": title,
                "block_content": block_content_text,
                "block_order": block_order
            })
            block_order += 1
        elif title and not block_content_text: # A title was found but no content followed before next title/EOF
             conceptual_blocks.append({ # Add block with title but empty content
                "doc_id": doc_id,
                "source": source,
                "original_type": original_type,
                "block_id": str(uuid.uuid4()),
                "concept_type": concept_type,
                "concept_name": title,
                "block_content": "", # Empty content
                "block_order": block_order
            })
             block_order += 1


    # If no conceptual blocks were found by patterns, but there's content,
    # treat the whole document as a single block.
    if not conceptual_blocks and parsed_content and parsed_content.strip():
        print(f"No specific conceptual blocks found by patterns in {source}. Treating entire document as one block.")
        conceptual_blocks.append({
            "doc_id": doc_id,
            "source": source,
            "original_type": original_type,
            "block_id": str(uuid.uuid4()),
            "concept_type": "full_document",
            "concept_name": f"Full content of {os.path.basename(source)}",
            "block_content": parsed_content.strip(),
            "block_order": 0 # Only one block
        })

    return conceptual_blocks


def tag_all_documents(parsed_docs_data: List[Dict]) -> List[Dict]:
    """
    Processes a list of parsed documents to identify and tag conceptual blocks.
    """
    all_blocks = []
    if not parsed_docs_data:
        print("tag_all_documents: No parsed document data provided.")
        return all_blocks

    for doc_data in parsed_docs_data:
        doc_id = doc_data.get("doc_id", str(uuid.uuid4())) # Generate ID if missing
        source = doc_data.get("source", "Unknown source")
        original_type = doc_data.get("original_type", "unknown")
        parsed_content = doc_data.get("parsed_content", "")

        if not parsed_content.strip():
            print(f"Skipping document {source} due to empty parsed content.")
            continue
            
        print(f"Identifying conceptual blocks for: {source} (Type: {original_type})")
        blocks = identify_conceptual_blocks_in_document(doc_id, source, original_type, parsed_content)
        if blocks:
            print(f"Found {len(blocks)} conceptual blocks in {source}.")
            all_blocks.extend(blocks)
        else:
            print(f"No conceptual blocks identified in {source}.")
            
    return all_blocks

if __name__ == '__main__':
    # Example Usage:
    sample_parsed_latex_content = """
§ Introduction to Linear Algebra
Linear algebra is a branch of mathematics concerned with vector spaces and linear mappings between such spaces.

§.§ Vector Spaces
A vector space is a collection of objects called vectors, which may be added together and multiplied by numbers, called scalars.

§.§.§ Axioms of a Vector Space
There are several axioms...

Some other text not under a specific subsubsection.

§ Another Main Section
Content here.
    """
    
    sample_doc_data_latex = {
        "doc_id": "sample_latex_doc",
        "source": "sample.tex",
        "original_type": "latex",
        "parsed_content": sample_parsed_latex_content
    }

    # Test with a document that might not have any matching patterns
    sample_plain_text_content = "This is a plain text document without any special section markers. It just contains paragraphs of text. Another paragraph follows here."
    sample_doc_data_plain = {
        "doc_id": "sample_plain_doc",
        "source": "plain.txt",
        "original_type": "text", # No patterns defined for "text" yet in LATEX_CONCEPT_PATTERNS
        "parsed_content": sample_plain_text_content
    }

    all_docs_to_tag = [sample_doc_data_latex, sample_doc_data_plain]
    
    print("--- Testing Concept Tagger ---")
    conceptual_blocks = tag_all_documents(all_docs_to_tag)

    if conceptual_blocks:
        print(f"\n--- Total Conceptual Blocks Found: {len(conceptual_blocks)} ---")
        for i, block in enumerate(conceptual_blocks):
            print(f"\nBlock {i+1}:")
            print(f"  Doc ID: {block['doc_id']}")
            print(f"  Source: {block['source']}")
            print(f"  Block ID: {block['block_id']}")
            print(f"  Concept Type: {block['concept_type']}")
            print(f"  Concept Name: {block['concept_name']}")
            print(f"  Content (first 100 chars): {block['block_content'][:100]}...")
            print(f"  Order: {block['block_order']}")
    else:
        print("No conceptual blocks were generated by the tagger.")

