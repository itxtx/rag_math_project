# src/data_ingestion/concept_tagger.py
import re

# Define patterns for LaTeX-specific concepts
# These patterns assume that the latex_parser has already somewhat cleaned up the text
# and preserved these structures.
LATEX_CONCEPT_PATTERNS = {
    "section": r"\\section\*?\{(.*?)\}",
    "subsection": r"\\subsection\*?\{(.*?)\}",
    "subsubsection": r"\\subsubsection\*?\{(.*?)\}",
    "paragraph_title": r"\\paragraph\*?\{(.*?)\}", # Captures the title, content follows
    # More complex environments:
    # Group 1: environment name (theorem, definition, etc.)
    # Group 2 (optional): name/title in brackets, e.g., [Pythagorean Theorem]
    # Group 3: content of the environment
    "environment": r"\\begin\{(theorem|definition|lemma|proof|example|remark|corollary|proposition|problem|solution)\}\s*(?:\[(.*?)\])?([\s\S]*?)\\end\{\1\}"
}

def tag_latex_concepts(parsed_content: str, source_path: str) -> list:
    """
    Identifies and tags concepts in parsed LaTeX content using regex.
    Outputs a list of conceptual blocks.
    """
    conceptual_blocks = []
    
    if not parsed_content.strip(): # Handle empty or whitespace-only content
        return []
    
    found_blocks = []

    # Find section-like titles first as they are usually single lines
    for concept_type, pattern in LATEX_CONCEPT_PATTERNS.items():
        if concept_type in ["section", "subsection", "subsubsection", "paragraph_title"]:
            for match in re.finditer(pattern, parsed_content):
                name = match.group(1).strip()
                found_blocks.append({
                    "start": match.start(),
                    "end": match.end(),
                    "concept_type": concept_type,
                    "concept_name": name,
                    "content": match.group(0) 
                })
        elif concept_type == "environment":
            for match in re.finditer(pattern, parsed_content, re.IGNORECASE): 
                env_type = match.group(1).lower()
                env_name = match.group(2).strip() if match.group(2) else None
                env_content = match.group(3).strip()
                found_blocks.append({
                    "start": match.start(),
                    "end": match.end(),
                    "concept_type": env_type, 
                    "concept_name": env_name, 
                    "content": env_content 
                })

    found_blocks.sort(key=lambda b: b["start"])

    last_block_end = 0
    for block in found_blocks:
        if block["start"] > last_block_end:
            general_text = parsed_content[last_block_end:block["start"]].strip()
            if general_text:
                conceptual_blocks.append({
                    "source": source_path,
                    "original_type": "latex",
                    "concept_type": "general_content",
                    "concept_name": None,
                    "content": general_text
                })
        
        conceptual_blocks.append({
            "source": source_path,
            "original_type": "latex",
            "concept_type": block["concept_type"],
            "concept_name": block.get("concept_name"),
            "content": block["content"]
        })
        last_block_end = block["end"]

    # Add any remaining general content after the last identified block OR if no blocks were found
    if last_block_end < len(parsed_content):
        general_text = parsed_content[last_block_end:].strip()
        if general_text:
            conceptual_blocks.append({
                "source": source_path,
                "original_type": "latex",
                "concept_type": "general_content",
                "concept_name": None,
                "content": general_text
            })
    
    # If the entire content was general and no specific blocks were found,
    # the above logic (last_block_end < len(parsed_content)) will correctly add it.
    # No need for the `if not found_blocks and parsed_content:` check.

    return conceptual_blocks


def tag_pdf_concepts(parsed_content: str, source_path: str) -> list:
    """
    Placeholder for identifying and tagging concepts in parsed PDF content.
    Currently treats the whole content as a single 'general_content' block.
    """
    conceptual_blocks = []
    if parsed_content.strip(): 
        conceptual_blocks.append({
            "source": source_path,
            "original_type": "pdf",
            "concept_type": "general_content",
            "concept_name": None,
            "content": parsed_content.strip()
        })
    return conceptual_blocks


def tag_concepts_in_document(parsed_doc_data: dict) -> list:
    """
    Processes a single parsed document's data to identify and tag concepts.
    """
    doc_type = parsed_doc_data.get("type")
    content = parsed_doc_data.get("content", "")
    source = parsed_doc_data.get("source", "unknown_source")

    if not content or not content.strip(): # Ensure content is not empty or just whitespace
        return []

    if doc_type == "latex":
        return tag_latex_concepts(content, source)
    elif doc_type == "pdf":
        return tag_pdf_concepts(content, source)
    else:
        print(f"Warning: Unknown document type '{doc_type}' for concept tagging. Treating as general content.")
        return [{
            "source": source,
            "original_type": doc_type,
            "concept_type": "general_content",
            "concept_name": None,
            "content": content.strip()
        }] if content.strip() else []


def tag_all_documents(parsed_docs_list: list) -> list:
    """
    Takes a list of parsed document data and returns a flat list of all conceptual blocks
    from all documents.
    """
    all_conceptual_blocks = []
    for doc_data in parsed_docs_list:
        blocks = tag_concepts_in_document(doc_data)
        all_conceptual_blocks.extend(blocks)
    return all_conceptual_blocks


if __name__ == '__main__':
    sample_latex_parsed_content = """
\\section{Introduction to Algebra}
Algebra is a branch of mathematics.
It involves variables like $x$ and $y$.

\\subsection{Basic Equations}
Consider the equation $x + 5 = 10$.
We can solve for $x$.

\\begin{definition}[Variable]
A variable is a symbol that represents a quantity that may vary.
\\end{definition}

Some more text here.

\\begin{theorem}[Fundamental Theorem of Algebra]
Every non-constant single-variable polynomial with complex coefficients has at least one complex root.
\\end{theorem}

This theorem is very important.
And some final text.
    """
    
    sample_pdf_parsed_content = """
This is a sample PDF document.
It discusses various topics including calculus and linear algebra.
Derivatives are a key concept in calculus.
Matrices are fundamental to linear algebra.
    """

    mock_parsed_docs = [
        {"source": "doc1.tex", "type": "latex", "content": sample_latex_parsed_content},
        {"source": "doc2.pdf", "type": "pdf", "content": sample_pdf_parsed_content},
        {"source": "doc3.tex", "type": "latex", "content": "Just some plain text in a tex file $a=b$."}
    ]

    print("--- Tagging concepts in all documents ---")
    all_blocks = tag_all_documents(mock_parsed_docs)

    for i, block in enumerate(all_blocks):
        print(f"\n--- Block {i+1} ---")
        print(f"  Source: {block['source']}")
        print(f"  Original Type: {block['original_type']}")
        print(f"  Concept Type: {block['concept_type']}")
        print(f"  Concept Name: {block['concept_name']}")
        print(f"  Content Snippet: {block['content'][:100].replace(chr(10), ' ')}...") 
