import pytest
from unittest.mock import patch, MagicMock, ANY
import uuid
import os
import numpy as np
import time
import weaviate
from sentence_transformers import SentenceTransformer

from src import config
from src.data_ingestion import vector_store_manager

# Mock SentenceTransformer setup
mock_st_model_instance_vsm = MagicMock(spec=SentenceTransformer)
mock_st_model_instance_vsm.get_sentence_embedding_dimension.return_value = 384
mock_st_model_instance_vsm.encode.return_value = np.array([0.1] * 384, dtype=np.float32)

mock_sentence_transformer_class_vsm = MagicMock(spec=SentenceTransformer)
mock_sentence_transformer_class_vsm.return_value = mock_st_model_instance_vsm

@pytest.fixture
def client_mock():
    """Create a mock Weaviate client with common setup."""
    client = MagicMock(spec=weaviate.Client)
    client.is_ready = MagicMock(return_value=True)
    client.schema = MagicMock()
    
    mock_404_response = MagicMock()
    mock_404_response.status_code = 404
    simulated_404_exception = weaviate.exceptions.UnexpectedStatusCodeException(
        message="Simulated Schema Not Found", response=mock_404_response
    )
    
    client.schema.get.side_effect = simulated_404_exception
    client.schema.exists = MagicMock(return_value=False)
    client.schema.create_class = MagicMock(return_value=None)
    client.schema.property = MagicMock()
    client.schema.property.create = MagicMock()
    
    batch_mock = MagicMock()
    batch_mock.configure = MagicMock(return_value=None)
    batch_mock.add_data_object = MagicMock(return_value=None)
    batch_mock.failed_objects = []
    
    client.batch = MagicMock()
    client.batch.__enter__ = MagicMock(return_value=batch_mock)
    client.batch.__exit__ = MagicMock(return_value=None)
    client.batch.failed_objects = []
    
    return client

@pytest.fixture
def mock_sentence_transformer():
    """Mock the SentenceTransformer class."""
    with patch('src.data_ingestion.vector_store_manager.SentenceTransformer', new=mock_sentence_transformer_class_vsm):
        yield mock_sentence_transformer_class_vsm

def test_get_weaviate_client_success(client_mock):
    """Test successful Weaviate client creation."""
    with patch('src.data_ingestion.vector_store_manager.weaviate.Client', return_value=client_mock):
        client = vector_store_manager.get_weaviate_client()
        assert client is not None
        assert client.is_ready.called

def test_get_weaviate_client_failure_then_success(client_mock):
    """Test Weaviate client creation with retries."""
    client_mock.is_ready.side_effect = [False, False, True]
    with patch('src.data_ingestion.vector_store_manager.weaviate.Client', return_value=client_mock), \
         patch('time.sleep', return_value=None):
        client = vector_store_manager.get_weaviate_client()
        assert client is not None
        assert client_mock.is_ready.call_count == 3

def test_create_weaviate_schema_new(client_mock):
    """Test creating a new Weaviate schema."""
    client_mock.schema.exists.return_value = False
    client_mock.schema.get.reset_mock()
    
    vector_store_manager.create_weaviate_schema(client_mock)
    
    assert client_mock.schema.exists.called_with(vector_store_manager.WEAVIATE_CLASS_NAME)
    assert not client_mock.schema.get.called
    assert client_mock.schema.create_class.called_once()

def test_create_weaviate_schema_exists(client_mock):
    """Test handling existing Weaviate schema."""
    client_mock.schema.exists.return_value = True
    client_mock.schema.get.side_effect = None
    client_mock.schema.get.return_value = {
        "class": vector_store_manager.WEAVIATE_CLASS_NAME,
        "properties": [
            {"name": p["name"]} for p in vector_store_manager.create_weaviate_schema.__defaults__[0]['properties']
        ] if vector_store_manager.create_weaviate_schema.__defaults__ else []
    }
    client_mock.schema.property.create.reset_mock()
    
    vector_store_manager.create_weaviate_schema(client_mock)
    
    assert client_mock.schema.exists.called_with(vector_store_manager.WEAVIATE_CLASS_NAME)
    assert client_mock.schema.get.called_with(vector_store_manager.WEAVIATE_CLASS_NAME)
    assert not client_mock.schema.create_class.called
    assert not client_mock.schema.property.create.called

def test_generate_standard_embedding(mock_sentence_transformer):
    """Test generating standard embeddings."""
    text = "This is a test sentence."
    embedding = vector_store_manager.generate_standard_embedding(text)
    assert embedding is not None
    assert len(embedding) == 384
    mock_st_model_instance_vsm.encode.assert_called_with(
        text, convert_to_tensor=False, normalize_embeddings=True
    )

def test_embed_chunk_data(mock_sentence_transformer):
    """Test embedding chunk data."""
    chunk_data = {"chunk_text": "Sample chunk text."}
    embedding = vector_store_manager.embed_chunk_data(chunk_data)
    assert embedding is not None
    assert len(embedding) == 384

def test_embed_chunk_data_empty_text(mock_sentence_transformer):
    """Test handling empty text in chunk data."""
    chunk_data = {"chunk_text": "   "}
    embedding = vector_store_manager.embed_chunk_data(chunk_data)
    assert embedding is None

def test_embed_and_store_chunks(client_mock):
    """Test embedding and storing chunks."""
    client_mock.schema.exists.return_value = False
    client_mock.schema.get.side_effect = client_mock.schema.get.side_effect
    
    dummy_chunks = [{
        "chunk_id": str(uuid.uuid4()),
        "doc_id": "d1",
        "source_path": "s1",
        "original_doc_type": "t1",
        "concept_type": "ct1",
        "concept_name": "cn1",
        "parent_block_id": "pb1",
        "chunk_text": "text1",
        "parent_block_content": "pb_content1",
        "sequence_in_block": 0,
        "filename": "f1.tex"
    }]
    
    vector_store_manager.embed_and_store_chunks(client_mock, dummy_chunks, batch_size=1)
    
    client_mock.batch.configure.assert_called_once_with(
        batch_size=1, dynamic=True, timeout_retries=3
    )
    assert client_mock.batch.__enter__().add_data_object.call_count == 1
    
    first_call_args = client_mock.batch.__enter__().add_data_object.call_args_list[0]
    assert first_call_args[1]['class_name'] == vector_store_manager.WEAVIATE_CLASS_NAME
    assert first_call_args[1]['uuid'] == dummy_chunks[0]['chunk_id']
    assert 'vector' in first_call_args[1]
    assert len(first_call_args[1]['vector']) == 384
    assert first_call_args[1]['data_object']['chunk_text'] == "text1"
    assert first_call_args[1]['data_object']['parent_block_id'] == "pb1"

def test_embed_and_store_chunks_embedding_error(client_mock):
    """Test handling embedding errors during chunk storage."""
    client_mock.schema.exists.return_value = False
    client_mock.schema.get.side_effect = client_mock.schema.get.side_effect
    
    with patch('src.data_ingestion.vector_store_manager.embed_chunk_data',
              side_effect=[np.array([0.2]*384).tolist(), None]) as mock_embed_func:
        dummy_chunks = [
            {
                "chunk_id": str(uuid.uuid4()),
                "chunk_text": "good text",
                "source_path": "s",
                "original_doc_type": "t",
                "concept_type": "c",
                "parent_block_content": "p",
                "sequence_in_block": 0,
                "parent_block_id": "pb_good",
                "doc_id": "doc_good"
            },
            {
                "chunk_id": str(uuid.uuid4()),
                "chunk_text": "bad text (simulated error)",
                "source_path": "s",
                "original_doc_type": "t",
                "concept_type": "c",
                "parent_block_content": "p",
                "sequence_in_block": 0,
                "parent_block_id": "pb_bad",
                "doc_id": "doc_bad"
            }
        ]
        
        vector_store_manager.embed_and_store_chunks(client_mock, dummy_chunks)
        
        assert mock_embed_func.call_count == 2
        assert client_mock.batch.__enter__().add_data_object.call_count == 1 