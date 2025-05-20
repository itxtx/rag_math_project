# tests/test_chunker.py
import unittest
import uuid
from src.data_ingestion.chunker import chunk_conceptual_blocks
# We don't need to import RecursiveCharacterTextSplitter here for testing the main function,
# unless we want to test its specific behavior in isolation (which is usually not needed for integration).

class TestChunker(unittest.TestCase):

    def test_chunk_conceptual_blocks_simple(self):
        """Test basic chunking of a single conceptual block."""
        conceptual_blocks = [
            {
                "doc_id": "doc1", "source": "source1.txt", "original_type": "text",
                "block_id": str(uuid.uuid4()), "concept_type": "section", "concept_name": "Intro",
                "block_content": "This is the first sentence. This is the second sentence. This is the third sentence, which is a bit longer. The fourth sentence follows."
            }
        ]
        # Expecting chunk_size=1000, chunk_overlap=150 by default from chunker.py if not overridden by app.py
        # For this test, let's use smaller values to force splitting.
        # The chunker.py demo uses chunk_size=100, overlap=20, min_chunk_size=10
        
        final_chunks = chunk_conceptual_blocks(
            conceptual_blocks, 
            chunk_size=50, 
            chunk_overlap=10,
            min_chunk_size=5 # Allow small chunks for this test
        )

        self.assertTrue(len(final_chunks) > 0, "Should produce at least one chunk.")
        
        for chunk in final_chunks:
            self.assertEqual(chunk["doc_id"], "doc1")
            self.assertEqual(chunk["concept_name"], "Intro")
            self.assertTrue(len(chunk["chunk_text"]) > 0)
            self.assertIn("parent_block_id", chunk)
            self.assertIn("parent_block_content", chunk)
            self.assertEqual(chunk["parent_block_content"], conceptual_blocks[0]["block_content"])

        # Example: if the first chunk is "This is the first sentence. This is the second sen" (length 50)
        # and second is "s is the second sentence. This is the third senten" (overlap "s is the second sen")
        # This specific assertion depends heavily on the RecursiveCharacterTextSplitter's behavior
        # with the default separators and the given text.
        if len(final_chunks) > 1:
            self.assertTrue(final_chunks[0]["chunk_text"].startswith("This is the first sentence."))
            # Check overlap (this is tricky to assert precisely without knowing exact split points)
            # For instance, the second chunk might start with part of the first chunk's end due to overlap.

    def test_chunk_multiple_blocks(self):
        """Test chunking with multiple conceptual blocks."""
        conceptual_blocks = [
            {
                "doc_id": "docA", "source": "docA.txt", "original_type": "text",
                "block_id": str(uuid.uuid4()), "concept_type": "section", "concept_name": "Section 1",
                "block_content": "Content for section 1. It is fairly long and should be split. More content to ensure splitting occurs."
            },
            {
                "doc_id": "docA", "source": "docA.txt", "original_type": "text",
                "block_id": str(uuid.uuid4()), "concept_type": "section", "concept_name": "Section 2",
                "block_content": "Content for section 2. This is shorter."
            }
        ]
        final_chunks = chunk_conceptual_blocks(conceptual_blocks, chunk_size=30, chunk_overlap=5, min_chunk_size=5)
        
        self.assertTrue(len(final_chunks) > 2, "Expected multiple chunks from multiple blocks.")
        
        doc_a_chunks = [c for c in final_chunks if c["doc_id"] == "docA"]
        self.assertEqual(len(doc_a_chunks), len(final_chunks))

        section_1_chunks = [c for c in doc_a_chunks if c["concept_name"] == "Section 1"]
        section_2_chunks = [c for c in doc_a_chunks if c["concept_name"] == "Section 2"]
        
        self.assertTrue(len(section_1_chunks) >= 2, "Section 1 should be split into at least 2 chunks.")
        self.assertTrue(len(section_2_chunks) >= 1, "Section 2 should produce at least 1 chunk.")

    def test_chunk_empty_or_short_block_content(self):
        """Test behavior with empty or very short block content."""
        conceptual_blocks = [
            {
                "doc_id": "docB", "source": "docB.txt", "original_type": "text",
                "block_id": str(uuid.uuid4()), "concept_type": "section", "concept_name": "Empty Section",
                "block_content": "   " # Whitespace only
            },
            {
                "doc_id": "docB", "source": "docB.txt", "original_type": "text",
                "block_id": str(uuid.uuid4()), "concept_type": "section", "concept_name": "Too Short Section",
                "block_content": "Short." 
            }
        ]
        # min_chunk_size is 10 in the chunker's demo, let's use that for consistency
        final_chunks = chunk_conceptual_blocks(conceptual_blocks, chunk_size=100, chunk_overlap=20, min_chunk_size=10)
        
        # Expect 0 chunks because "Short." (6 chars) < min_chunk_size (10)
        # and whitespace only is also skipped.
        self.assertEqual(len(final_chunks), 0, "Should produce no chunks from empty or too short content.")

    def test_min_chunk_size_filter(self):
        conceptual_blocks = [
            {
                "doc_id": "docC", "source": "docC.txt", "original_type": "text",
                "block_id": str(uuid.uuid4()), "concept_type": "section", "concept_name": "Test Min Size",
                "block_content": "This is just long enough. This part is not." # First sentence is 28 chars.
            }
        ]
        # Splitter might make "This is just long enough." and " This part is not."
        # If chunk_size is small, e.g., 20, overlap 5.
        # "This is just long e" (len 20)
        # "st long enough. Th" (len 20) -> after strip: "st long enough. Th"
        # "nough. This part i" (len 20) -> after strip: "nough. This part i"
        # "s part is not." (len 14) -> after strip: "s part is not."
        
        # Test with min_chunk_size that would filter some out
        final_chunks = chunk_conceptual_blocks(conceptual_blocks, chunk_size=25, chunk_overlap=5, min_chunk_size=20)
        self.assertTrue(len(final_chunks) > 0)
        for chunk in final_chunks:
            self.assertTrue(len(chunk["chunk_text"]) >= 20)

        # Test where all chunks might be filtered
        final_chunks_all_filtered = chunk_conceptual_blocks(conceptual_blocks, chunk_size=15, chunk_overlap=5, min_chunk_size=30)
        self.assertEqual(len(final_chunks_all_filtered), 0)


if __name__ == '__main__':
    unittest.main()
