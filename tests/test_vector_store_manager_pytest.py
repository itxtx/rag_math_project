import pytest
from unittest.mock import MagicMock, patch, ANY
import httpx
from src.data_ingestion import vector_store_manager

# Mock the weaviate client before it's imported by other modules
mock_weaviate_client = MagicMock()
mock_weaviate_client.is_ready.return_value = True
mock_weaviate_client.schema.exists.return_value = True
mock_weaviate_client.schema.create_class.return_value = None
mock_weaviate_client.batch.configure.return_value = None
mock_weaviate_client.batch.__enter__.return_value = mock_weaviate_client.batch
mock_weaviate_client.batch.__exit__.return_value = None
mock_weaviate_client.batch.add_data_object.return_value = None

# It's crucial to patch 'weaviate.connect_to_local' which is used in the source code
@patch('weaviate.connect_to_local', return_value=mock_weaviate_client)
def test_get_weaviate_client_success(mock_connect):
    """Test successful connection to Weaviate."""
    mock_client = MagicMock()
    mock_client.is_ready.return_value = True
    mock_connect.return_value = mock_client
    
    client = vector_store_manager.get_weaviate_client()
    
    mock_connect.assert_called_once()
    assert client is not None

@patch('weaviate.connect_to_local')
def test_get_weaviate_client_failure_then_success(mock_connect):
    """Test reconnection logic."""
    # Simulate failure on first call, success on second
    mock_connect.side_effect = [Exception("Connection failed"), mock_weaviate_client]
    
    with pytest.raises(Exception, match="Connection failed"):
        vector_store_manager.get_weaviate_client()

    # The second call should succeed
    client = vector_store_manager.get_weaviate_client()
    assert mock_connect.call_count == 2
    assert client is not None

# Helper to create a mock httpx.Response for exceptions
def create_mock_response(status_code, json_body):
    mock_res = MagicMock(spec=httpx.Response)
    mock_res.status_code = status_code
    mock_res.json.return_value = json_body
    return mock_res

@patch('weaviate.connect_to_local')
def test_create_weaviate_schema_new(mock_connect):
    """Test schema creation when it does not exist."""
    mock_client = MagicMock()
    mock_client.collections.exists.return_value = False
    mock_client.collections.create_from_dict.return_value = None
    mock_connect.return_value = mock_client
    
    vector_store_manager.create_weaviate_schema(mock_client)
    
    mock_client.collections.exists.assert_called_with("MathConcept")
    mock_client.collections.create_from_dict.assert_called_once()

@patch('weaviate.connect_to_local')
def test_create_weaviate_schema_exists(mock_connect):
    """Test schema creation when it already exists."""
    mock_client = MagicMock()
    mock_client.collections.exists.return_value = True
    mock_connect.return_value = mock_client
    
    vector_store_manager.create_weaviate_schema(mock_client)
    
    mock_client.collections.exists.assert_called_with("MathConcept")
    mock_client.collections.create_from_dict.assert_not_called()

@patch('src.data_ingestion.vector_store_manager.SentenceTransformer')
@patch('weaviate.connect_to_local')
@pytest.mark.asyncio
async def test_fast_embed_and_store_chunks(mock_connect, mock_transformer):
    """Test the optimized embedding and storage function."""
    mock_client = MagicMock()
    mock_connect.return_value = mock_client
    
    mock_embedding_model = MagicMock()
    mock_embedding_model.encode.return_value = [[0.1, 0.2, 0.3]]
    mock_transformer.return_value = mock_embedding_model
    
    chunks = [{"chunk_id": "1", "chunk_text": "This is a test chunk."}]
    
    await vector_store_manager.fast_embed_and_store_chunks(mock_client, chunks)
    
    mock_client.batch.configure.assert_called_once()
    mock_client.batch.__enter__.assert_called()

@patch('src.data_ingestion.vector_store_manager.SentenceTransformer')
@patch('weaviate.connect_to_local')
@pytest.mark.asyncio
async def test_fast_embed_and_store_chunks_embedding_error(mock_connect, mock_transformer):
    """Test handling of embedding errors in the optimized function."""
    mock_client = MagicMock()
    mock_connect.return_value = mock_client
    
    mock_embedding_model = MagicMock()
    mock_embedding_model.encode.side_effect = Exception("Embedding failed")
    mock_transformer.return_value = mock_embedding_model
    
    chunks = [{"chunk_id": "1", "chunk_text": "This is a test chunk."}]
    
    await vector_store_manager.fast_embed_and_store_chunks(mock_client, chunks)
    
    # When an error occurs, no data should be added
    mock_client.batch.__enter__().add_data_object.assert_not_called()

@patch('src.data_ingestion.vector_store_manager.SentenceTransformer')
def test_generate_standard_embedding(mock_transformer):
    """Test the standard embedding generation function."""
    mock_embedding_model = MagicMock()
    mock_embedding_model.encode.return_value = [0.1, 0.2, 0.3]
    mock_transformer.return_value = mock_embedding_model
    
    embedding = vector_store_manager.generate_standard_embedding("Test text")
    
    mock_embedding_model.encode.assert_called_once_with(
        "Test text",
        convert_to_tensor=False,
        normalize_embeddings=True
    )
    assert embedding == [0.1, 0.2, 0.3]

def test_embed_chunk_data():
    """Test embedding generation for a chunk."""
    with patch('src.data_ingestion.vector_store_manager.generate_standard_embedding') as mock_generate:
        mock_generate.return_value = [0.1, 0.2, 0.3]
        
        chunk = {"chunk_text": "Test chunk text"}
        embedding = vector_store_manager.embed_chunk_data(chunk)
        
        mock_generate.assert_called_once_with("Test chunk text")
        assert embedding == [0.1, 0.2, 0.3]

def test_embed_chunk_data_empty():
    """Test embedding generation for an empty chunk."""
    chunk = {"chunk_text": ""}
    embedding = vector_store_manager.embed_chunk_data(chunk)
    assert embedding is None

def test_get_weaviate_client_success():
    with patch('src.data_ingestion.vector_store_manager.weaviate.connect_to_local', autospec=True) as mock_connect:
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_connect.return_value = mock_client
        client = vector_store_manager.get_weaviate_client()
        mock_connect.assert_called_once()
        assert client is not None

def test_get_weaviate_client_failure_then_success():
    with patch('src.data_ingestion.vector_store_manager.weaviate.connect_to_local', autospec=True) as mock_connect:
        mock_connect.side_effect = [Exception("Connection failed"), MagicMock(is_ready=MagicMock(return_value=True))]
        try:
            vector_store_manager.get_weaviate_client()
        except Exception as e:
            assert str(e) == "Connection failed"
        # The second call should succeed
        client = vector_store_manager.get_weaviate_client()
        assert mock_connect.call_count == 2
        assert client is not None

def test_create_weaviate_schema_new():
    mock_client = MagicMock()
    mock_client.collections.exists.return_value = False
    mock_client.collections.create_from_dict.return_value = None
    vector_store_manager.create_weaviate_schema(mock_client)
    mock_client.collections.exists.assert_called_with("MathConcept")
    mock_client.collections.create_from_dict.assert_called_once()

def test_create_weaviate_schema_exists():
    mock_client = MagicMock()
    mock_client.collections.exists.return_value = True
    vector_store_manager.create_weaviate_schema(mock_client)
    mock_client.collections.exists.assert_called_with("MathConcept")
    mock_client.collections.create_from_dict.assert_not_called()