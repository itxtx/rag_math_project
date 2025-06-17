import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import uuid
import asyncio
import time
import numpy as np
import os

from src.retrieval.retriever import HybridRetriever
from src.generation.question_generator_rag import RAGQuestionGenerator
from src.data_ingestion import vector_store_manager
from src import config
import weaviate
from sentence_transformers import SentenceTransformer

# Test data constants
TEST_CHUNK_1_ID = str(uuid.uuid4())
TEST_CHUNK_1_TEXT = "The first law of thermodynamics, also known as the law of conservation of energy, states that energy cannot be created or destroyed in an isolated system. It can only be transformed from one form to another."
TEST_CHUNK_1_CONCEPT = "First Law of Thermodynamics"
TEST_CHUNK_1_PARENT_BLOCK_ID = "pb_thermo_law1"
TEST_CHUNK_1_DOC_ID = "doc_thermo"

TEST_CHUNK_2_ID = str(uuid.uuid4())
TEST_CHUNK_2_TEXT = "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy, through a process that uses sunlight, water, and carbon dioxide."
TEST_CHUNK_2_CONCEPT = "Photosynthesis"
TEST_CHUNK_2_PARENT_BLOCK_ID = "pb_photosynthesis"
TEST_CHUNK_2_DOC_ID = "doc_bio"

DUMMY_CHUNKS_FOR_INTEGRATION_TEST = [
    {
        "chunk_id": TEST_CHUNK_1_ID,
        "doc_id": TEST_CHUNK_1_DOC_ID,
        "source_path": "thermo_doc.tex",
        "original_doc_type": "latex",
        "concept_type": "scientific_law",
        "concept_name": TEST_CHUNK_1_CONCEPT,
        "chunk_text": TEST_CHUNK_1_TEXT,
        "parent_block_id": TEST_CHUNK_1_PARENT_BLOCK_ID,
        "parent_block_content": f"\\section{{{TEST_CHUNK_1_CONCEPT}}}\n{TEST_CHUNK_1_TEXT}",
        "sequence_in_block": 0,
        "filename": "thermo_doc.tex"
    },
    {
        "chunk_id": TEST_CHUNK_2_ID,
        "doc_id": TEST_CHUNK_2_DOC_ID,
        "source_path": "biology_notes.pdf",
        "original_doc_type": "pdf",
        "concept_type": "biological_process",
        "concept_name": TEST_CHUNK_2_CONCEPT,
        "chunk_text": TEST_CHUNK_2_TEXT,
        "parent_block_id": TEST_CHUNK_2_PARENT_BLOCK_ID,
        "parent_block_content": f"Chapter on {TEST_CHUNK_2_CONCEPT}.\n{TEST_CHUNK_2_TEXT}",
        "sequence_in_block": 0,
        "filename": "biology_notes.pdf"
    }
]

# Mock setup for SentenceTransformer
@pytest.fixture
def mock_sentence_transformer():
    mock_st_model_instance = MagicMock(spec=SentenceTransformer)
    mock_st_model_instance.get_sentence_embedding_dimension.return_value = 384
    mock_embedding_vector = np.array([0.1] * 383 + [0.2], dtype=np.float32)
    mock_st_model_instance.encode.return_value = mock_embedding_vector

    mock_sentence_transformer_class = MagicMock(spec=SentenceTransformer)
    mock_sentence_transformer_class.return_value = mock_st_model_instance

    return mock_sentence_transformer_class

@pytest.fixture
def weaviate_client():
    """Fixture to set up and tear down Weaviate client for tests."""
    if not hasattr(config, 'WEAVIATE_URL') or not config.WEAVIATE_URL:
        pytest.skip("WEAVIATE_URL not configured.")

    client = vector_store_manager.get_weaviate_client()
    
    # Clean up existing schema if it exists
    try:
        if client.schema.exists(vector_store_manager.WEAVIATE_CLASS_NAME):
            client.schema.delete_class(vector_store_manager.WEAVIATE_CLASS_NAME)
            time.sleep(1)
    except Exception as e:
        pytest.skip(f"Could not delete existing class: {e}")

    # Create fresh schema
    vector_store_manager.create_weaviate_schema(client)

    # Ingest test data
    vector_store_manager.embed_and_store_chunks(
        client,
        DUMMY_CHUNKS_FOR_INTEGRATION_TEST,
        batch_size=2
    )
    
    # Wait for indexing
    time.sleep(5)

    yield client

    # Cleanup after tests
    try:
        if client.schema.exists(vector_store_manager.WEAVIATE_CLASS_NAME):
            client.schema.delete_class(vector_store_manager.WEAVIATE_CLASS_NAME)
    except Exception as e:
        print(f"Error during cleanup: {e}")

@pytest.fixture
def retriever(weaviate_client):
    """Fixture to create a HybridRetriever instance."""
    return HybridRetriever(weaviate_client=weaviate_client)

@pytest.fixture
def question_generator():
    """Fixture to create a RAGQuestionGenerator instance with mocked LLM."""
    generator = RAGQuestionGenerator()
    mock_llm_response_text = "1. What does the first law of thermodynamics state about energy?\n2. Can energy be created or destroyed according to this law?"
    generator._call_llm_api = AsyncMock(return_value=mock_llm_response_text)
    return generator

@pytest.mark.asyncio
@patch('src.data_ingestion.vector_store_manager.SentenceTransformer')
async def test_retrieve_and_generate_questions_flow(
    mock_st_class,
    mock_sentence_transformer,
    weaviate_client,
    retriever,
    question_generator
):
    """Test the complete flow of retrieving chunks and generating questions."""
    # Set up the mock
    mock_st_class.return_value = mock_sentence_transformer.return_value

    # Verify Weaviate client
    assert weaviate_client is not None, "Weaviate client not initialized"
    assert retriever.client is not None, "Retriever client not initialized"

    # Test retrieval
    query = "energy conservation law"
    retrieved_chunks = retriever.search(
        query_text=query,
        search_type="semantic",
        limit=1,
        certainty=0.01
    )

    assert retrieved_chunks is not None
    assert len(retrieved_chunks) > 0, f"Should retrieve at least one chunk for query '{query}'"
    
    retrieved_texts = [chunk.get("chunk_text", "") for chunk in retrieved_chunks]
    assert any(TEST_CHUNK_1_TEXT in text for text in retrieved_texts), \
        f"Expected text related to '{TEST_CHUNK_1_CONCEPT}' in retrieved chunks. Actual: {retrieved_texts}"

    # Test question generation
    num_questions_to_generate = 2
    generated_questions = await question_generator.generate_questions(
        context_chunks=retrieved_chunks,
        num_questions=num_questions_to_generate,
        question_type="factual",
        difficulty_level="intermediate"
    )

    assert generated_questions is not None
    assert len(generated_questions) == num_questions_to_generate

    expected_questions = [
        "What does the first law of thermodynamics state about energy?",
        "Can energy be created or destroyed according to this law?"
    ]
    assert generated_questions == expected_questions

    # Verify LLM call
    question_generator._call_llm_api.assert_called_once()
    call_args = question_generator._call_llm_api.call_args
    assert call_args is not None
    
    prompt_sent_to_llm = call_args[0][0]
    assert any(ret_text in prompt_sent_to_llm for ret_text in retrieved_texts if ret_text), \
        "The prompt sent to LLM should contain text from a retrieved chunk"
    assert f"Generate exactly {num_questions_to_generate} question(s)" in prompt_sent_to_llm
    assert "factual questions" in prompt_sent_to_llm 