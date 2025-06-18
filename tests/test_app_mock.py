import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from contextlib import asynccontextmanager

@asynccontextmanager 
async def mock_lifespan(app):
    """Mock lifespan that doesn't initialize real components"""
    app.state.components = MagicMock()
    app.state.profile_manager = MagicMock() 
    app.state.question_selector = MagicMock()
    app.state.active_interactions = {}
    yield

@pytest.fixture
def mocked_app():
    """Provide completely mocked FastAPI app"""
    with patch('src.data_ingestion.vector_store_manager.get_weaviate_client'), \
         patch('src.learner_model.profile_manager.LearnerProfileManager'), \
         patch('sentence_transformers.SentenceTransformer'):
        from src.api.fast_api import app
        app.router.lifespan_context = mock_lifespan
        return app 