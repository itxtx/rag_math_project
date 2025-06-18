import pytest
from unittest.mock import MagicMock, patch, ANY
import httpx

# Mock the weaviate client before it's imported by other modules
mock_weaviate_client = MagicMock()

# It's crucial to patch 'weaviate.connect_to_local' which is used in the source code
@patch('weaviate.connect_to_local', return_value=mock_weaviate_client)
def test_get_weaviate_client_success(mock_connect):
    """Test successful connection to Weaviate."""
    from src.data_ingestion.vector_store_manager import VectorStoreManager
    manager = VectorStoreManager()
    client = manager.get_weaviate_client()
    mock_connect.assert_called_once()
    assert client is not None

@patch('weaviate.connect_to_local')
def test_get_weaviate_client_failure_then_success(mock_connect):
    """Test reconnection logic."""
    from src.data_ingestion.vector_store_manager import VectorStoreManager
    # Simulate failure on first call, success on second
    mock_connect.side_effect = [Exception("Connection failed"), mock_weaviate_client]
    
    manager = VectorStoreManager()
    
    with pytest.raises(Exception, match="Connection failed"):
        manager.get_weaviate_client()

    # The second call should succeed
    client = manager.get_weaviate_client()
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
    from src.data_ingestion.vector_store_manager import VectorStoreManager, UnexpectedStatusCodeError
    
    mock_client = MagicMock()
    # Simulate schema not found error
    mock_client.schema.get.side_effect = UnexpectedStatusCodeError(
        "Simulated Schema Not Found",
        create_mock_response(404, {"error": "schema not found"})
    )
    mock_connect.return_value = mock_client
    
    manager = VectorStoreManager()
    manager.create_weaviate_schema()
    
    mock_client.schema.get.assert_called_with("Embeddings")
    mock_client.schema.create.assert_called_once()


@patch('weaviate.connect_to_local')
def test_create_weaviate_schema_exists(mock_connect):
    """Test schema creation when it already exists."""
    from src.data_ingestion.vector_store_manager import VectorStoreManager
    mock_client = MagicMock()
    # Simulate schema already exists
    mock_client.schema.get.return_value = {"name": "Embeddings"}
    mock_connect.return_value = mock_client
    
    manager = VectorStoreManager()
    manager.create_weaviate_schema()
    
    mock_client.schema.get.assert_called_with("Embeddings")
    mock_client.schema.create.assert_not_called()


@patch('src.data_ingestion.vector_store_manager.SentenceTransformer')
@patch('weaviate.connect_to_local')
def test_embed_and_store_chunks(mock_connect, mock_transformer):
    """Test embedding and storing text chunks."""
    from src.data_ingestion.vector_store_manager import VectorStoreManager
    
    mock_client = MagicMock()
    mock_connect.return_value = mock_client
    
    # Configure the mock SentenceTransformer
    mock_embedding_model = MagicMock()
    mock_embedding_model.encode.return_value = [[0.1, 0.2, 0.3]]
    mock_transformer.return_value = mock_embedding_model
    
    manager = VectorStoreManager()
    chunks = [{"chunk_id": "1", "content": "This is a test chunk."}]
    manager.embed_and_store_chunks(chunks)
    
    mock_client.batch.configure.assert_called_once()
    mock_client.batch.add_data_object.assert_called_once()
    args, kwargs = mock_client.batch.add_data_object.call_args
    assert kwargs['uuid'] == "1"
    assert kwargs['vector'] == [0.1, 0.2, 0.3]


@patch('src.data_ingestion.vector_store_manager.SentenceTransformer')
@patch('weaviate.connect_to_local')
def test_embed_and_store_chunks_embedding_error(mock_connect, mock_transformer):
    """Test handling of embedding errors."""
    from src.data_ingestion.vector_store_manager import VectorStoreManager
    
    mock_client = MagicMock()
    mock_connect.return_value = mock_client
    
    # Configure the mock SentenceTransformer to raise an error
    mock_embedding_model = MagicMock()
    mock_embedding_model.encode.side_effect = Exception("Embedding failed")
    mock_transformer.return_value = mock_embedding_model
    
    manager = VectorStoreManager()
    chunks = [{"chunk_id": "1", "content": "This is a test chunk."}]
    
    with pytest.raises(Exception, match="Embedding failed"):
        manager.embed_and_store_chunks(chunks)
        
    mock_client.batch.add_data_object.assert_not_called()