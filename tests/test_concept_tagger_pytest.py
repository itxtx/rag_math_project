import pytest
import os
from src.data_ingestion.concept_tagger import identify_conceptual_blocks_in_document, tag_all_documents, LATEX_CONCEPT_PATTERNS
import regex as re

def test_identify_conceptual_blocks_latex_sections():
    doc_id = "test_latex_doc_1"
    source = "test_latex.tex"
    original_type = "latex"
    parsed_content = """
§ Introduction
This is the first part.

§.§ Background
More details here.
Some text.

§.§.§ Specific Point
A very specific detail.

§ Another Section
Content of the second section.
    """
    expected_blocks = [
        {"concept_type": "section", "concept_name": "Introduction", "block_content": "This is the first part."},
        {"concept_type": "subsection", "concept_name": "Background", "block_content": "More details here.\nSome text."},
        {"concept_type": "subsubsection", "concept_name": "Specific Point", "block_content": "A very specific detail."},
        {"concept_type": "section", "concept_name": "Another Section", "block_content": "Content of the second section."}
    ]
    result_blocks = identify_conceptual_blocks_in_document(doc_id, source, original_type, parsed_content)
    assert len(result_blocks) == len(expected_blocks)
    for i, expected in enumerate(expected_blocks):
        assert result_blocks[i]["concept_type"] == expected["concept_type"]
        assert result_blocks[i]["concept_name"] == expected["concept_name"]
        assert result_blocks[i]["block_content"].strip() == expected["block_content"].strip()
        assert result_blocks[i]["doc_id"] == doc_id
        assert result_blocks[i]["source"] == source
        assert result_blocks[i]["original_type"] == original_type
        assert "block_id" in result_blocks[i]
        assert result_blocks[i]["block_order"] == i

def test_identify_conceptual_blocks_no_patterns_match():
    doc_id = "test_plain_doc_1"
    source = "plain.txt"
    original_type = "text"
    parsed_content = "This is a simple text document. It has several sentences but no section markers that match LATEX_CONCEPT_PATTERNS."
    expected_blocks = [{
        "concept_type": "full_document",
        "concept_name": f"Full content of {os.path.basename(source)}",
        "block_content": parsed_content.strip()
    }]
    result_blocks = identify_conceptual_blocks_in_document(doc_id, source, original_type, parsed_content)
    assert len(result_blocks) == 1
    assert result_blocks[0]["concept_type"] == expected_blocks[0]["concept_type"]
    assert result_blocks[0]["concept_name"] == expected_blocks[0]["concept_name"]
    assert result_blocks[0]["block_content"] == expected_blocks[0]["block_content"]

def test_identify_conceptual_blocks_empty_content():
    doc_id = "empty_doc"
    source = "empty.txt"
    original_type = "text"
    parsed_content = "   "
    result_blocks = identify_conceptual_blocks_in_document(doc_id, source, original_type, parsed_content)
    assert len(result_blocks) == 0

def test_tag_all_documents_multiple_docs():
    parsed_docs_data = [
        {"doc_id": "latex_doc_A", "source": "docA.tex", "original_type": "latex", "parsed_content": "§ Section Alpha\nContent Alpha.\n§.§ Sub Alpha One\nSub content."},
        {"doc_id": "plain_doc_B", "source": "docB.txt", "original_type": "text", "parsed_content": "Just plain text here."},
        {"doc_id": "latex_doc_C_empty", "source": "docC.tex", "original_type": "latex", "parsed_content": ""}
    ]
    all_blocks = tag_all_documents(parsed_docs_data)
    assert len(all_blocks) == 3
    doc_a_blocks = [b for b in all_blocks if b["doc_id"] == "latex_doc_A"]
    assert len(doc_a_blocks) == 2
    assert doc_a_blocks[0]["concept_name"] == "Section Alpha"
    assert doc_a_blocks[0]["concept_type"] == "section"
    assert doc_a_blocks[1]["concept_name"] == "Sub Alpha One"
    assert doc_a_blocks[1]["concept_type"] == "subsection"
    doc_b_blocks = [b for b in all_blocks if b["doc_id"] == "plain_doc_B"]
    assert len(doc_b_blocks) == 1
    assert doc_b_blocks[0]["concept_type"] == "full_document"

def test_latex_patterns_capture_titles():
    section_text = "§ My Section Title  \nContent"
    match = re.search(LATEX_CONCEPT_PATTERNS["section"], section_text, re.MULTILINE)
    assert match is not None
    assert match.group(1).strip() == "My Section Title"
    subsection_text = "§.§ My Subsection Title\nContent"
    match = re.search(LATEX_CONCEPT_PATTERNS["subsection"], subsection_text, re.MULTILINE)
    assert match is not None
    assert match.group(1).strip() == "My Subsection Title"
    subsubsection_text = "§.§.§ My SubSub Title \nContent"
    match = re.search(LATEX_CONCEPT_PATTERNS["subsubsection"], subsubsection_text, re.MULTILINE)
    assert match is not None
    assert match.group(1).strip() == "My SubSub Title" 