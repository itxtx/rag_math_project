import unittest
from src.data_ingestion.concept_tagger import tag_latex_concepts, tag_pdf_concepts, tag_concepts_in_document

class TestConceptTagger(unittest.TestCase):

    def test_tag_latex_concepts_sections_and_environments(self):
        latex_content = """
\\section{Introduction}
This is the intro.
\\subsection{Background}
Some background info.
\\begin{theorem}[Pythagorean Theorem]
For a right-angled triangle, $a^2 + b^2 = c^2$.
\\end{theorem}
More text after theorem.
\\begin{definition}
A prime number is a natural number greater than 1.
\\end{definition}
        """
        blocks = tag_latex_concepts(latex_content, "test.tex")
        
        self.assertTrue(len(blocks) >= 5) # Intro, General, Background, General, Theorem, General, Definition

        section_block = next((b for b in blocks if b["concept_type"] == "section"), None)
        self.assertIsNotNone(section_block)
        self.assertEqual(section_block["concept_name"], "Introduction")
        self.assertEqual(section_block["content"], "\\section{Introduction}")

        subsection_block = next((b for b in blocks if b["concept_type"] == "subsection"), None)
        self.assertIsNotNone(subsection_block)
        self.assertEqual(subsection_block["concept_name"], "Background")

        theorem_block = next((b for b in blocks if b["concept_type"] == "theorem"), None)
        self.assertIsNotNone(theorem_block)
        self.assertEqual(theorem_block["concept_name"], "Pythagorean Theorem")
        self.assertIn("a^2 + b^2 = c^2", theorem_block["content"])

        definition_block = next((b for b in blocks if b["concept_type"] == "definition"), None)
        self.assertIsNotNone(definition_block)
        self.assertIsNone(definition_block["concept_name"]) # No name provided in []
        self.assertIn("A prime number is a natural number", definition_block["content"])

        general_content_blocks = [b for b in blocks if b["concept_type"] == "general_content"]
        self.assertTrue(any("This is the intro." in b["content"] for b in general_content_blocks))
        self.assertTrue(any("More text after theorem." in b["content"] for b in general_content_blocks))


    def test_tag_latex_concepts_only_general_content(self):
        latex_content = "This is just some plain text. With an equation $E=mc^2$."
        blocks = tag_latex_concepts(latex_content, "test.tex")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["concept_type"], "general_content")
        self.assertEqual(blocks[0]["content"], latex_content)

    def test_tag_latex_concepts_empty_content(self):
        blocks = tag_latex_concepts("", "test.tex")
        self.assertEqual(len(blocks), 0)

    def test_tag_pdf_concepts_simple(self):
        pdf_content = "This is content from a PDF. It has sentences."
        blocks = tag_pdf_concepts(pdf_content, "test.pdf")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["concept_type"], "general_content")
        self.assertEqual(blocks[0]["content"], pdf_content)
        self.assertEqual(blocks[0]["original_type"], "pdf")

    def test_tag_pdf_concepts_empty_content(self):
        blocks = tag_pdf_concepts("", "test.pdf")
        self.assertEqual(len(blocks), 0)

    def test_tag_concepts_in_document_latex(self):
        doc_data = {
            "source": "mydoc.tex", 
            "type": "latex", 
            "content": "\\section{My Section}\nSome content."
        }
        blocks = tag_concepts_in_document(doc_data)
        self.assertTrue(len(blocks) > 0)
        self.assertEqual(blocks[0]["original_type"], "latex")
        self.assertEqual(blocks[0]["concept_type"], "section")

    def test_tag_concepts_in_document_pdf(self):
        doc_data = {
            "source": "mydoc.pdf", 
            "type": "pdf", 
            "content": "PDF text."
        }
        blocks = tag_concepts_in_document(doc_data)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["original_type"], "pdf")
        self.assertEqual(blocks[0]["concept_type"], "general_content")

    def test_tag_latex_no_ending_general_content(self):
        latex_content = "\\section{Only Section}"
        blocks = tag_latex_concepts(latex_content, "test.tex")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["concept_type"], "section")
        self.assertEqual(blocks[0]["content"], "\\section{Only Section}")
        
    def test_tag_latex_paragraph_command(self):
        latex_content = """
\\paragraph{My Paragraph Title}
This is the content of the paragraph.
        """
        blocks = tag_latex_concepts(latex_content, "test.tex")
        paragraph_title_block = next((b for b in blocks if b["concept_type"] == "paragraph_title"), None)
        self.assertIsNotNone(paragraph_title_block)
        self.assertEqual(paragraph_title_block["concept_name"], "My Paragraph Title")

        general_block_after = next((b for b in blocks if b["concept_type"] == "general_content" and "content of the paragraph" in b["content"]), None)
        self.assertIsNotNone(general_block_after)

if __name__ == '__main__':
    unittest.main()
