# tests/test_concept_tagger.py
import unittest
import os # For os.path.basename if used in assertions
from src.data_ingestion.concept_tagger import identify_conceptual_blocks_in_document, tag_all_documents, LATEX_CONCEPT_PATTERNS
import regex as re
class TestConceptTagger(unittest.TestCase):

    def test_identify_conceptual_blocks_latex_sections(self):
        """
        Tests identification of sections, subsections, etc., from parsed LaTeX content.
        """
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
            {
                "concept_type": "section", 
                "concept_name": "Introduction", 
                "block_content": "This is the first part." # Content until next section marker
            },
            {
                "concept_type": "subsection", 
                "concept_name": "Background", 
                "block_content": "More details here.\nSome text."
            },
            {
                "concept_type": "subsubsection", 
                "concept_name": "Specific Point", 
                "block_content": "A very specific detail." # Content until next section marker (Another Section)
            },
            {
                "concept_type": "section",
                "concept_name": "Another Section",
                "block_content": "Content of the second section."
            }
        ]

        result_blocks = identify_conceptual_blocks_in_document(doc_id, source, original_type, parsed_content)
        
        self.assertEqual(len(result_blocks), len(expected_blocks), "Number of identified blocks does not match expected.")

        for i, expected in enumerate(expected_blocks):
            self.assertEqual(result_blocks[i]["concept_type"], expected["concept_type"], f"Block {i} concept_type mismatch.")
            self.assertEqual(result_blocks[i]["concept_name"], expected["concept_name"], f"Block {i} concept_name mismatch.")
            # Normalize whitespace for block content comparison as leading/trailing spaces can vary
            self.assertEqual(result_blocks[i]["block_content"].strip(), expected["block_content"].strip(), f"Block {i} block_content mismatch.")
            self.assertEqual(result_blocks[i]["doc_id"], doc_id)
            self.assertEqual(result_blocks[i]["source"], source)
            self.assertEqual(result_blocks[i]["original_type"], original_type)
            self.assertIn("block_id", result_blocks[i]) # Check for presence of block_id
            self.assertEqual(result_blocks[i]["block_order"], i)


    def test_identify_conceptual_blocks_no_patterns_match(self):
        """
        Tests behavior when no specific patterns match, should treat as full document.
        """
        doc_id = "test_plain_doc_1"
        source = "plain.txt"
        original_type = "text" # Assuming no patterns are defined for "text" type
        parsed_content = "This is a simple text document. It has several sentences but no section markers that match LATEX_CONCEPT_PATTERNS."
        
        expected_blocks = [
            {
                "concept_type": "full_document",
                "concept_name": f"Full content of {os.path.basename(source)}",
                "block_content": parsed_content.strip()
            }
        ]
        
        result_blocks = identify_conceptual_blocks_in_document(doc_id, source, original_type, parsed_content)
        
        self.assertEqual(len(result_blocks), 1)
        self.assertEqual(result_blocks[0]["concept_type"], expected_blocks[0]["concept_type"])
        self.assertEqual(result_blocks[0]["concept_name"], expected_blocks[0]["concept_name"])
        self.assertEqual(result_blocks[0]["block_content"], expected_blocks[0]["block_content"])

    def test_identify_conceptual_blocks_empty_content(self):
        """
        Tests behavior with empty parsed content.
        """
        doc_id = "empty_doc"
        source = "empty.txt"
        original_type = "text"
        parsed_content = "   " # Whitespace only
        
        result_blocks = identify_conceptual_blocks_in_document(doc_id, source, original_type, parsed_content)
        self.assertEqual(len(result_blocks), 0, "Should return no blocks for empty or whitespace-only content.")

    def test_tag_all_documents_multiple_docs(self):
        """
        Tests the tag_all_documents function with a mix of documents.
        """
        parsed_docs_data = [
            {
                "doc_id": "latex_doc_A", "source": "docA.tex", "original_type": "latex",
                "parsed_content": "§ Section Alpha\nContent Alpha.\n§.§ Sub Alpha One\nSub content."
            },
            {
                "doc_id": "plain_doc_B", "source": "docB.txt", "original_type": "text",
                "parsed_content": "Just plain text here."
            },
            {
                "doc_id": "latex_doc_C_empty", "source": "docC.tex", "original_type": "latex",
                "parsed_content": "" # Empty content
            }
        ]

        all_blocks = tag_all_documents(parsed_docs_data)

        # Expected counts:
        # docA.tex: section "Section Alpha", subsection "Sub Alpha One" -> 2 blocks
        # docB.txt: "full_document" -> 1 block
        # docC.tex: empty -> 0 blocks
        self.assertEqual(len(all_blocks), 3, "Total number of blocks from all documents is incorrect.")

        # Check types and names for some blocks
        doc_a_blocks = [b for b in all_blocks if b["doc_id"] == "latex_doc_A"]
        self.assertEqual(len(doc_a_blocks), 2)
        self.assertEqual(doc_a_blocks[0]["concept_name"], "Section Alpha")
        self.assertEqual(doc_a_blocks[0]["concept_type"], "section")
        self.assertEqual(doc_a_blocks[1]["concept_name"], "Sub Alpha One")
        self.assertEqual(doc_a_blocks[1]["concept_type"], "subsection")

        doc_b_blocks = [b for b in all_blocks if b["doc_id"] == "plain_doc_B"]
        self.assertEqual(len(doc_b_blocks), 1)
        self.assertEqual(doc_b_blocks[0]["concept_type"], "full_document")
        
    def test_latex_patterns_capture_titles(self):
        """
        Specifically tests if the LATEX_CONCEPT_PATTERNS capture titles correctly.
        """
        section_text = "§ My Section Title  \nContent"
        match = re.search(LATEX_CONCEPT_PATTERNS["section"], section_text, re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1).strip(), "My Section Title")

        subsection_text = "§.§ My Subsection Title\nContent"
        match = re.search(LATEX_CONCEPT_PATTERNS["subsection"], subsection_text, re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1).strip(), "My Subsection Title")

        subsubsection_text = "§.§.§ My SubSub Title \nContent"
        match = re.search(LATEX_CONCEPT_PATTERNS["subsubsection"], subsubsection_text, re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1).strip(), "My SubSub Title")

if __name__ == '__main__':
    unittest.main()
