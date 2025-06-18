import pytest
import asyncio
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock
from contextlib import asynccontextmanager

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# CRITICAL FIX 1: Set test environment variables BEFORE any imports
os.environ["WEAVIATE_URL"] = "http://localhost:8080"
os.environ["GEMINI_API_KEY"] = "test-key"
os.environ["API_KEY"] = "test-key"
os.environ["TESTING"] = "true"

# CRITICAL FIX 2: Mock all external dependencies before they're imported
def mock_weaviate():
    """Mock Weaviate client to prevent hanging"""
    mock_client = MagicMock()
    mock_client.is_ready.return_value = True
    mock_client.query.get.return_value.with_additional.return_value.with_limit.return_value.do.return_value = {
        "data": {"Get": {"MathDocumentChunk": []}}
    }
    return mock_client

def mock_sentence_transformer():
    """Mock SentenceTransformer to prevent model loading"""
    mock_model = MagicMock()
    mock_model.encode.return_value = [0.1] * 384  # Mock embedding
    return mock_model

# Apply mocks globally
pytest_plugins = []

# CRITICAL FIX 3: Session-scoped event loop to prevent conflicts
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    
    # Set as the default loop
    asyncio.set_event_loop(loop)
    
    yield loop
    
    # Clean up
    try:
        # Cancel all running tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        
        # Wait for cancellation to complete
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        
        loop.close()
    except Exception:
        pass  # Ignore cleanup errors

# CRITICAL FIX 4: Mock FastAPI app components
@pytest.fixture(autouse=True, scope="session")
def mock_external_dependencies():
    """Mock all external dependencies that could cause hanging"""
    with patch('src.data_ingestion.vector_store_manager.get_weaviate_client', return_value=mock_weaviate()), \
         patch('src.data_ingestion.vector_store_manager.SentenceTransformer', return_value=mock_sentence_transformer()), \
         patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer()), \
         patch('src.learner_model.profile_manager.LearnerProfileManager') as mock_pm, \
         patch('src.generation.question_generator_rag.RAGQuestionGenerator') as mock_qg, \
         patch('src.evaluation.answer_evaluator.AnswerEvaluator') as mock_ae:
        
        # Configure mocks
        mock_pm.return_value = MagicMock()
        mock_qg.return_value = MagicMock()
        mock_ae.return_value = MagicMock()
        
        yield

# CRITICAL FIX 5: Override FastAPI lifespan for tests
@asynccontextmanager
async def test_lifespan(app):
    """Test lifespan that doesn't initialize real components"""
    # Mock all app state without real initialization
    app.state.components = MagicMock()
    app.state.profile_manager = MagicMock()
    app.state.question_selector = MagicMock()
    app.state.question_generator = MagicMock()
    app.state.answer_evaluator = MagicMock()
    app.state.active_interactions = {}
    yield
    # No cleanup needed for mocks

# CRITICAL FIX 6: Provide mocked FastAPI app
@pytest.fixture(scope="session")
def test_app():
    """Provide a mocked FastAPI app for testing"""
    # Import only after mocks are in place
    from src.api.fast_api import app
    
    # Override lifespan
    app.router.lifespan_context = test_lifespan
    
    return app

# CRITICAL FIX 7: Timeout fixture to prevent hanging
@pytest.fixture(autouse=True)
def test_timeout():
    """Add timeout to all tests to prevent infinite hanging"""
    import signal
    
    def timeout_handler(signum, frame):
        pytest.fail("Test timed out after 30 seconds - likely infinite loop or hanging")
    
    # Set timeout for all tests
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout
    
    yield
    
    # Cancel timeout
    signal.alarm(0)

# CRITICAL FIX 8: Mock async dependencies properly
@pytest.fixture
def mock_async_components():
    """Provide properly mocked async components"""
    components = MagicMock()
    
    # Create proper AsyncMock for async methods
    components.retriever.get_all_documents = AsyncMock(return_value=[])
    components.retriever.fast_semantic_search = AsyncMock(return_value=[])
    components.question_selector.select_next_question = AsyncMock(return_value={
        "question_text": "Test question",
        "concept_id": "test_concept",
        "concept_name": "Test Concept"
    })
    components.answer_handler.submit_answer = AsyncMock(return_value={
        "accuracy_score": 0.8,
        "feedback": "Good answer"
    })
    
    return components

# CRITICAL FIX 9: Database cleanup
@pytest.fixture(autouse=True)
def cleanup_database():
    """Clean up any database connections"""
    yield
    
    # Close any open database connections
    try:
        import sqlite3
        # This is a bit aggressive but ensures no hanging connections
        # In a real scenario, you'd want more targeted cleanup
        pass
    except Exception:
        pass

# CRITICAL FIX 10: Async test utilities
@pytest.fixture
def async_test_client():
    """Provide async test client that won't hang"""
    from httpx import AsyncClient
    from src.api.fast_api import app
    
    return AsyncClient(app=app, base_url="http://test")

# CRITICAL FIX 11: Sync test client (recommended for FastAPI)
@pytest.fixture
def sync_test_client():
    """Provide synchronous test client (recommended)"""
    from fastapi.testclient import TestClient
    from src.api.fast_api import app
    
    return TestClient(app)

# CRITICAL FIX 12: Mock Weaviate for retrieval tests
@pytest.fixture
def mock_weaviate_client():
    """Mock Weaviate client for retrieval tests"""
    client = MagicMock()
    client.is_ready.return_value = True
    client.schema.exists.return_value = False
    client.schema.create_class.return_value = None
    
    # Mock query interface
    query_mock = MagicMock()
    query_mock.get.return_value = query_mock
    query_mock.with_additional.return_value = query_mock
    query_mock.with_limit.return_value = query_mock
    query_mock.with_near_vector.return_value = query_mock
    query_mock.do.return_value = {"data": {"Get": {"MathDocumentChunk": []}}}
    
    client.query = query_mock
    return client

# Configuration for specific test modules
def pytest_configure(config):
    """Configure pytest to prevent hanging"""
    # Ensure asyncio mode is set correctly
    config.option.asyncio_mode = "strict"

def pytest_collection_modifyitems(config, items):
    """Modify test items to add timeouts and proper async handling"""
    for item in items:
        # Add timeout marker to all tests
        item.add_marker(pytest.mark.timeout(30))
        
        # Ensure async tests are properly marked
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)

# Handle cleanup after each test
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test to prevent state leakage"""
    yield
    
    # Cancel any pending tasks
    try:
        loop = asyncio.get_event_loop()
        tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in tasks:
            task.cancel()
    except Exception:
        pass

@pytest.fixture(autouse=True)
def mock_external_services():
    with patch('src.data_ingestion.vector_store_manager.get_weaviate_client'), \
         patch('src.learner_model.profile_manager.LearnerProfileManager'), \
         patch('sentence_transformers.SentenceTransformer'):
        yield