import pytest
import uuid
from src.data_ingestion.chunker import chunk_conceptual_blocks

@pytest.fixture
def simple_conceptual_block():
    """Create a simple conceptual block for testing."""
    return [{
        "doc_id": "doc1",
        "source": "source1.txt",
        "original_type": "text",
        "block_id": str(uuid.uuid4()),
        "concept_type": "section",
        "concept_name": "Intro",
        "block_content": "This is the first sentence. This is the second sentence. This is the third sentence, which is a bit longer. The fourth sentence follows."
    }]

@pytest.fixture
def multiple_conceptual_blocks():
    """Create multiple conceptual blocks for testing."""
    return [
        {
            "doc_id": "docA",
            "source": "docA.txt",
            "original_type": "text",
            "block_id": str(uuid.uuid4()),
            "concept_type": "section",
            "concept_name": "Section 1",
            "block_content": "Content for section 1. It is fairly long and should be split. More content to ensure splitting occurs."
        },
        {
            "doc_id": "docA",
            "source": "docA.txt",
            "original_type": "text",
            "block_id": str(uuid.uuid4()),
            "concept_type": "section",
            "concept_name": "Section 2",
            "block_content": "Content for section 2. This is shorter."
        }
    ]

@pytest.fixture
def empty_or_short_blocks():
    """Create blocks with empty or short content."""
    return [
        {
            "doc_id": "docB",
            "source": "docB.txt",
            "original_type": "text",
            "block_id": str(uuid.uuid4()),
            "concept_type": "section",
            "concept_name": "Empty Section",
            "block_content": "   "  # Whitespace only
        },
        {
            "doc_id": "docB",
            "source": "docB.txt",
            "original_type": "text",
            "block_id": str(uuid.uuid4()),
            "concept_type": "section",
            "concept_name": "Too Short Section",
            "block_content": "Short."
        }
    ]

def test_chunk_conceptual_blocks_simple(simple_conceptual_block):
    """Test basic chunking of a single conceptual block."""
    final_chunks = chunk_conceptual_blocks(
        simple_conceptual_block,
        chunk_size=50,
        chunk_overlap=10,
        min_chunk_size=5  # Allow small chunks for this test
    )

    assert len(final_chunks) > 0, "Should produce at least one chunk."
    
    for chunk in final_chunks:
        assert chunk["doc_id"] == "doc1"
        assert chunk["concept_name"] == "Intro"
        assert len(chunk["chunk_text"]) > 0
        assert "parent_block_id" in chunk
        assert "parent_block_content" in chunk
        assert chunk["parent_block_content"] == simple_conceptual_block[0]["block_content"]

    if len(final_chunks) > 1:
        assert final_chunks[0]["chunk_text"].startswith("This is the first sentence.")

def test_chunk_multiple_blocks(multiple_conceptual_blocks):
    """Test chunking with multiple conceptual blocks."""
    final_chunks = chunk_conceptual_blocks(
        multiple_conceptual_blocks,
        chunk_size=30,
        chunk_overlap=5,
        min_chunk_size=5
    )
    
    assert len(final_chunks) > 2, "Expected multiple chunks from multiple blocks."
    
    doc_a_chunks = [c for c in final_chunks if c["doc_id"] == "docA"]
    assert len(doc_a_chunks) == len(final_chunks)

    section_1_chunks = [c for c in doc_a_chunks if c["concept_name"] == "Section 1"]
    section_2_chunks = [c for c in doc_a_chunks if c["concept_name"] == "Section 2"]
    
    assert len(section_1_chunks) >= 2, "Section 1 should be split into at least 2 chunks."
    assert len(section_2_chunks) >= 1, "Section 2 should produce at least 1 chunk."

def test_chunk_empty_or_short_block_content(empty_or_short_blocks):
    """Test behavior with empty or very short block content."""
    final_chunks = chunk_conceptual_blocks(
        empty_or_short_blocks,
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_size=10
    )
    
    assert len(final_chunks) == 0, "Should produce no chunks from empty or too short content."

def test_min_chunk_size_filter():
    """Test filtering based on minimum chunk size."""
    conceptual_blocks = [{
        "doc_id": "docC",
        "source": "docC.txt",
        "original_type": "text",
        "block_id": str(uuid.uuid4()),
        "concept_type": "section",
        "concept_name": "Test Min Size",
        "block_content": "This is just long enough. This part is not."
    }]
    
    # Test with min_chunk_size that would filter some out
    final_chunks = chunk_conceptual_blocks(
        conceptual_blocks,
        chunk_size=25,
        chunk_overlap=5,
        min_chunk_size=20
    )
    assert len(final_chunks) > 0
    for chunk in final_chunks:
        assert len(chunk["chunk_text"]) >= 20

    # Test where all chunks might be filtered
    final_chunks_all_filtered = chunk_conceptual_blocks(
        conceptual_blocks,
        chunk_size=15,
        chunk_overlap=5,
        min_chunk_size=30
    )
    assert len(final_chunks_all_filtered) == 0 