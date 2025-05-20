# tests/test_chunker.py
import unittest
from src.data_ingestion.chunker import chunk_conceptual_blocks, chunk_content

class TestChunker(unittest.TestCase):

    def test_chunk_content_simple_split(self):
        text = "This is a test sentence that is longer than the chunk size."
        chunks = chunk_content(text, chunk_size=20, chunk_overlap=5)
        self.assertTrue(len(chunks) > 1)
        self.assertEqual(chunks[0], "This is a test sente")
        # Expected overlap: "sente"
        # Next chunk starts at index 15 (20-5)
        self.assertEqual(chunks[1], "sentence that is lon") 

    def test_chunk_content_no_split_needed(self):
        text = "Short text."
        chunks = chunk_content(text, chunk_size=20, chunk_overlap=5)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_chunk_content_exact_multiple(self):
        text = "abcd efgh ijkl mnop" # 19 chars
        chunks = chunk_content(text, chunk_size=9, chunk_overlap=1) # 9-1 = 8 step
        # "abcd efgh" (9)
        # "h ijkl mn" (9) - starts at index 8
        # "nop" (3) - starts at index 16
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "abcd efgh")
        self.assertEqual(chunks[1], "h ijkl mn")
        self.assertEqual(chunks[2], "op") # Corrected: last chunk will be shorter if not full

    def test_chunk_content_with_overlap(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        chunk_size = 10
        chunk_overlap = 3
        chunks = chunk_content(text, chunk_size, chunk_overlap)
        
        self.assertEqual(chunks[0], "abcdefghij")
        self.assertEqual(chunks[1], "hijklmnopq") # Starts at index 7 (10-3)
        self.assertEqual(chunks[2], "opqrstuvwx") # Starts at index 14 (7+7)
        self.assertEqual(chunks[3], "uvwxyz")     # Starts at index 21 (14+7)

    def test_chunk_content_empty_text(self):
        chunks = chunk_content("", 100, 20)
        self.assertEqual(len(chunks), 0)

    def test_chunk_conceptual_blocks_small_block(self):
        conceptual_blocks = [{
            "source": "test.tex", "original_type": "latex",
            "concept_type": "definition", "concept_name": "My Def",
            "content": "This is a short definition."
        }]
        final_chunks = chunk_conceptual_blocks(conceptual_blocks, chunk_size=100, chunk_overlap=20)
        self.assertEqual(len(final_chunks), 1)
        self.assertEqual(final_chunks[0]["chunk_text"], "This is a short definition.")
        self.assertEqual(final_chunks[0]["concept_type"], "definition")
        self.assertEqual(final_chunks[0]["sequence_in_block"], 0)

    def test_chunk_conceptual_blocks_large_block_needs_splitting(self):
        long_content = "This is a very long piece of general content that definitely needs to be split into multiple smaller chunks. " * 10
        conceptual_blocks = [{
            "source": "test.pdf", "original_type": "pdf",
            "concept_type": "general_content", "concept_name": None,
            "content": long_content
        }]
        final_chunks = chunk_conceptual_blocks(conceptual_blocks, chunk_size=100, chunk_overlap=20)
        self.assertTrue(len(final_chunks) > 1)
        for i, chunk in enumerate(final_chunks):
            self.assertEqual(chunk["concept_type"], "general_content")
            self.assertEqual(chunk["source"], "test.pdf")
            self.assertTrue(len(chunk["chunk_text"]) <= 100)
            self.assertEqual(chunk["sequence_in_block"], i)
            self.assertEqual(chunk["parent_block_content"], long_content)

    def test_chunk_conceptual_blocks_empty_content_block(self):
        conceptual_blocks = [{
            "source": "test.tex", "original_type": "latex",
            "concept_type": "section", "concept_name": "Empty Section",
            "content": "  " # Whitespace only
        }]
        final_chunks = chunk_conceptual_blocks(conceptual_blocks, chunk_size=100, chunk_overlap=20)
        self.assertEqual(len(final_chunks), 0)

    def test_chunk_conceptual_blocks_multiple_types(self):
        conceptual_blocks = [
            {"source": "doc.tex", "original_type": "latex", "concept_type": "section", "content": "\\section{S1}"},
            {"source": "doc.tex", "original_type": "latex", "concept_type": "general_content", "content": "Short general content."},
            {"source": "doc.tex", "original_type": "latex", "concept_type": "theorem", "content": "A" * 150} # Needs splitting
        ]
        final_chunks = chunk_conceptual_blocks(conceptual_blocks, chunk_size=50, chunk_overlap=10)
        
        self.assertEqual(final_chunks[0]["concept_type"], "section")
        self.assertEqual(final_chunks[0]["chunk_text"], "\\section{S1}")
        
        self.assertEqual(final_chunks[1]["concept_type"], "general_content")
        self.assertEqual(final_chunks[1]["chunk_text"], "Short general content.")
        
        # Theorem chunks
        theorem_chunks = [c for c in final_chunks if c["concept_type"] == "theorem"]
        self.assertTrue(len(theorem_chunks) > 1) # Expecting 150 / (50-10) approx
        self.assertEqual(theorem_chunks[0]["chunk_text"], "A" * 50)
        self.assertEqual(theorem_chunks[0]["sequence_in_block"], 0)
        self.assertEqual(theorem_chunks[1]["chunk_text"], "A" * 50) # Overlap makes this tricky to assert precisely without running the logic
        self.assertTrue(theorem_chunks[1]["chunk_text"].startswith("A" * 10)) # Check overlap
        self.assertEqual(theorem_chunks[1]["sequence_in_block"], 1)


    def test_min_chunk_size_threshold(self):
        conceptual_blocks = [{
            "source": "test.tex", "original_type": "latex",
            "concept_type": "definition", "concept_name": "Tiny Def",
            "content": "Tiny." # Shorter than MIN_CHUNK_SIZE_THRESHOLD (default 20)
        }]
        # Default MIN_CHUNK_SIZE_THRESHOLD is 20. "Tiny." is 5 chars.
        # It should still create a chunk because it's > 0 and the block itself isn't huge.
        final_chunks = chunk_conceptual_blocks(conceptual_blocks, chunk_size=100, chunk_overlap=20)
        self.assertEqual(len(final_chunks), 1)
        self.assertEqual(final_chunks[0]["chunk_text"], "Tiny.")

        conceptual_blocks_empty = [{
            "source": "test.tex", "original_type": "latex",
            "concept_type": "definition", "concept_name": "Empty Def",
            "content": "   " # Whitespace only
        }]
        final_chunks_empty = chunk_conceptual_blocks(conceptual_blocks_empty, chunk_size=100, chunk_overlap=20)
        self.assertEqual(len(final_chunks_empty), 0)


if __name__ == '__main__':
    unittest.main()