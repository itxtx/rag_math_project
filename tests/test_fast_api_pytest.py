import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException
import json
import os
from contextlib import asynccontextmanager

from src.api.fast_api import app, FastRAGComponents, get_api_key, API_KEY, get_fast_components
from src.api.models import (
    LearnerInteractionStartRequest,
    AnswerSubmissionRequest,
    QuestionResponse,
    AnswerSubmissionResponse,
    EvaluationResult,
    TopicResponse
)

# Set a dummy API key for testing
os.environ["API_KEY"] = "test-key"

def override_get_api_key():
    return API_KEY

app.dependency_overrides[get_api_key] = override_get_api_key

# Test API key for testing
TEST_API_KEY = "test-api-key-12345"

def override_get_fast_components():
    components = MagicMock()
    components.retriever = AsyncMock()
    components.question_selector = AsyncMock()
    components.answer_handler = AsyncMock()
    return components

app.dependency_overrides[get_fast_components] = override_get_fast_components

# --- Fixtures ---

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture
def mock_fast_components():
    """Fixture for mocking the FastRAGComponents dependency."""
    with patch('src.api.fast_api.FastRAGComponents') as mock:
        instance = mock.return_value
        instance.retriever = AsyncMock()
        instance.question_selector = AsyncMock()
        instance.answer_handler = AsyncMock()
        yield instance

@pytest.fixture
def mock_question_response():
    """Mock question response data."""
    return {
        "question_id": "test_question_123",
        "doc_id": "test_doc_456",
        "concept_name": "Test Concept",
        "question_text": "What is the derivative of x^2?",
        "context_for_evaluation": "The derivative of x^2 is 2x.",
        "is_new_concept_context_presented": False
    }

@pytest.fixture
def mock_evaluation_result():
    """Mock evaluation result data."""
    return {
        "accuracy_score": 0.8,
        "feedback": "Good answer! The derivative of x^2 is indeed 2x.",
        "correct_answer": "2x"
    }

@pytest.fixture
def mock_topic_response():
    """Mock topic response data."""
    return {
        "doc_id": "test_doc_456",
        "title": "Calculus Fundamentals",
        "description": "Introduction to calculus concepts",
        "concepts": ["derivatives", "limits", "integration"]
    }

# --- Helper Functions ---

def get_auth_headers():
    """Get headers with API key for authenticated requests."""
    return {"X-API-Key": TEST_API_KEY}

# --- Authentication Tests ---

def test_api_key_required(client):
    """Test that endpoints are protected by API key."""
    # Temporarily remove the override to test the protection
    app.dependency_overrides = {}
    
    response = client.get("/api/v1/health")
    assert response.status_code == 403
    
    # Restore the override for other tests
    app.dependency_overrides[get_api_key] = override_get_api_key

def test_health_check_success(client):
    """Test successful health check."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.2.0"}

# --- Topics Endpoint Tests ---

@pytest.mark.asyncio
async def test_list_topics_success(client, mock_fast_components):
    """Test successful topics listing."""
    app.state.fast_rag_components = mock_fast_components
    mock_fast_components.retriever.get_all_chunks_metadata.return_value = [
        {"doc_id": "doc1", "concept_name": "Concept 1", "text": "Text 1"}
    ]
    
    response = client.get("/api/v1/topics")
    
    assert response.status_code == 200
    assert len(response.json()) > 0

@pytest.mark.asyncio
async def test_list_topics_empty(client, mock_fast_components):
    """Test topics listing when no topics are available."""
    with patch('src.api.fast_api.get_fast_components', return_value=mock_fast_components):
        mock_fast_components.retriever.get_all_chunks_metadata.return_value = []
        
        response = client.get("/api/v1/topics", headers=get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

@pytest.mark.asyncio
async def test_list_topics_retriever_error(client, mock_fast_components):
    """Test topics listing when retriever raises an exception."""
    with patch('src.api.fast_api.get_fast_components', return_value=mock_fast_components):
        mock_fast_components.retriever.get_all_chunks_metadata.side_effect = Exception("Database error")
        
        response = client.get("/api/v1/topics", headers=get_auth_headers())
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

# --- Start Interaction Tests ---

@pytest.mark.asyncio
async def test_start_interaction_success(client, mock_fast_components, mock_question_response):
    """Test successful interaction start."""
    with patch('src.api.fast_api.get_fast_components', return_value=mock_fast_components):
        mock_fast_components.question_selector.select_next_question.return_value = mock_question_response
        
        response = client.post(
            "/api/v1/interaction/start",
            params={"learner_id": "123", "topic_id": "test_doc_456"},
            headers=get_auth_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["question_id"] == mock_question_response["question_id"]
        assert data["question_text"] == mock_question_response["question_text"]

@pytest.mark.asyncio
async def test_start_interaction_selector_error(client, mock_fast_components):
    """Test start interaction when question selector raises an exception."""
    with patch('src.api.fast_api.get_fast_components', return_value=mock_fast_components):
        mock_fast_components.question_selector.select_next_question.side_effect = Exception("Selection error")
        
        response = client.post(
            "/api/v1/interaction/start",
            params={"learner_id": "123"},
            headers=get_auth_headers()
        )
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

@pytest.mark.asyncio
async def test_start_interaction_no_questions_available(client, mock_fast_components):
    """Test start interaction when no questions are available."""
    with patch('src.api.fast_api.get_fast_components', return_value=mock_fast_components):
        mock_fast_components.question_selector.select_next_question.return_value = {
            "error": "No curriculum content available."
        }
        
        response = client.post(
            "/api/v1/interaction/start",
            params={"learner_id": "123"},
            headers=get_auth_headers()
        )
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

# --- Submit Answer Tests ---

@pytest.mark.asyncio
async def test_submit_answer_success(client, mock_fast_components, mock_evaluation_result):
    """Test successful answer submission."""
    with patch('src.api.fast_api.get_fast_components', return_value=mock_fast_components):
        mock_fast_components.answer_handler.process_answer.return_value = {
            "learner_id": "123",
            "question_id": "test_question_123",
            "doc_id": "test_doc_456",
            "evaluation": mock_evaluation_result
        }
        
        answer_data = {
            "learner_id": "123",
            "question_id": "test_question_123",
            "doc_id": "test_doc_456",
            "question_text": "What is the derivative of x^2?",
            "context_for_evaluation": "The derivative of x^2 is 2x.",
            "learner_answer": "2x"
        }
        
        response = client.post(
            "/api/v1/interaction/submit_answer",
            json=answer_data,
            headers=get_auth_headers()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["learner_id"] == "123"
        assert data["question_id"] == "test_question_123"
        assert "evaluation" in data

def test_submit_answer_invalid_request_missing_fields(client):
    """Test answer submission with missing required fields."""
    answer_data = {
        "learner_id": "123"
        # Missing other required fields
    }
    
    response = client.post(
        "/api/v1/interaction/submit_answer",
        json=answer_data,
        headers=get_auth_headers()
    )
    assert response.status_code == 422  # Validation error

def test_submit_answer_invalid_request_wrong_types(client):
    """Test answer submission with wrong data types."""
    answer_data = {
        "learner_id": 123,  # Should be string
        "question_id": "test_question_123",
        "doc_id": "test_doc_456",
        "question_text": "What is the derivative of x^2?",
        "context_for_evaluation": "The derivative of x^2 is 2x.",
        "learner_answer": "2x"
    }
    
    response = client.post(
        "/api/v1/interaction/submit_answer",
        json=answer_data,
        headers=get_auth_headers()
    )
    assert response.status_code == 422  # Validation error

@pytest.mark.asyncio
async def test_submit_answer_handler_error(client, mock_fast_components):
    """Test answer submission when answer handler raises an exception."""
    with patch('src.api.fast_api.get_fast_components', return_value=mock_fast_components):
        mock_fast_components.answer_handler.process_answer.side_effect = Exception("Processing error")
        
        answer_data = {
            "learner_id": "123",
            "question_id": "test_question_123",
            "doc_id": "test_doc_456",
            "question_text": "What is the derivative of x^2?",
            "context_for_evaluation": "The derivative of x^2 is 2x.",
            "learner_answer": "2x"
        }
        
        response = client.post(
            "/api/v1/interaction/submit_answer",
            json=answer_data,
            headers=get_auth_headers()
        )
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

# --- Performance Endpoint Tests ---

def test_performance_stats_success(client, mock_fast_components):
    """Test successful performance stats retrieval."""
    with patch('src.api.fast_api.get_fast_components', return_value=mock_fast_components):
        response = client.get("/api/v1/performance", headers=get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert "startup_time" in data
        assert "prewarmed" in data
        assert "components_ready" in data

# --- Clear Cache Tests ---

@pytest.mark.asyncio
async def test_clear_cache_success(client, mock_fast_components):
    """Test successful cache clearing."""
    with patch('src.api.fast_api.get_fast_components', return_value=mock_fast_components):
        mock_fast_components.retriever.clear_cache = AsyncMock()
        
        response = client.post("/api/v1/clear_cache", headers=get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Cache cleared successfully" in data["message"]

@pytest.mark.asyncio
async def test_clear_cache_error(client, mock_fast_components):
    """Test cache clearing when retriever raises an exception."""
    with patch('src.api.fast_api.get_fast_components', return_value=mock_fast_components):
        mock_fast_components.retriever.clear_cache = AsyncMock(side_effect=Exception("Cache error"))
        
        response = client.post("/api/v1/clear_cache", headers=get_auth_headers())
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

# --- Error Handling Tests ---

@pytest.mark.asyncio
async def test_components_dependency_injection_error(client):
    """Test error handling when component dependency injection fails."""
    with patch('src.api.fast_api.get_fast_components', side_effect=Exception("Dependency error")):
        response = client.get("/api/v1/health", headers=get_auth_headers())
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

def test_invalid_json_request(client):
    """Test handling of invalid JSON in request body."""
    response = client.post(
        "/api/v1/interaction/submit_answer",
        data="invalid json",
        headers={**get_auth_headers(), "Content-Type": "application/json"}
    )
    assert response.status_code == 422

# --- Edge Cases ---

@pytest.mark.asyncio
async def test_start_interaction_with_none_topic_id(client, mock_fast_components, mock_question_response):
    """Test start interaction with None topic_id."""
    with patch('src.api.fast_api.get_fast_components', return_value=mock_fast_components):
        mock_fast_components.question_selector.select_next_question.return_value = mock_question_response
        
        response = client.post(
            "/api/v1/interaction/start",
            params={"learner_id": "123", "topic_id": None},
            headers=get_auth_headers()
        )
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_submit_answer_with_empty_strings(client, mock_fast_components, mock_evaluation_result):
    """Test answer submission with empty strings in required fields."""
    with patch('src.api.fast_api.get_fast_components', return_value=mock_fast_components):
        mock_fast_components.answer_handler.process_answer.return_value = {
            "learner_id": "123",
            "question_id": "test_question_123",
            "doc_id": "test_doc_456",
            "evaluation": mock_evaluation_result
        }
        
        answer_data = {
            "learner_id": "",
            "question_id": "",
            "doc_id": "",
            "question_text": "",
            "context_for_evaluation": "",
            "learner_answer": ""
        }
        
        response = client.post(
            "/api/v1/interaction/submit_answer",
            json=answer_data,
            headers=get_auth_headers()
        )
        # Should handle empty strings gracefully (might be 200 or 422 depending on validation)
        assert response.status_code in [200, 422] 