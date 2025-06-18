import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.adaptive_engine.question_selector import QuestionSelector
from src.retrieval.retriever import HybridRetriever
from src.generation.question_generator_rag import RAGQuestionGenerator
from src.learner_model.profile_manager import LearnerProfileManager

# --- Fixtures ---

@pytest.fixture
def mock_retriever():
    """Fixture for a mocked HybridRetriever."""
    retriever = MagicMock(spec=HybridRetriever)
    # Mock the correct methods used by QuestionSelector
    retriever.get_all_chunks_metadata.return_value = []
    retriever.get_chunks_for_parent_block.return_value = []
    return retriever

@pytest.fixture
def mock_qg_rag():
    """Fixture for a mocked RAGQuestionGenerator."""
    # Use AsyncMock for async methods
    qg = AsyncMock(spec=RAGQuestionGenerator)
    qg.generate_questions.return_value = ["Generated question?"]
    return qg

@pytest.fixture
def mock_profile_manager():
    """Fixture for a mocked LearnerProfileManager."""
    pm = MagicMock(spec=LearnerProfileManager)
    pm.get_concepts_for_review.return_value = []
    pm.get_concept_knowledge.return_value = None
    return pm

@pytest.fixture
def question_selector(mock_retriever, mock_qg_rag, mock_profile_manager):
    """Fixture to create a QuestionSelector instance with mocked dependencies."""
    return QuestionSelector(
        profile_manager=mock_profile_manager,
        retriever=mock_retriever,
        question_generator=mock_qg_rag
    )

# --- Test Cases ---

@pytest.mark.asyncio
async def test_initialization_and_curriculum_map_loading(mock_retriever, mock_qg_rag, mock_profile_manager):
    """Test that QuestionSelector initializes and loads the curriculum map."""
    mock_retriever.get_all_chunks_metadata.return_value = [
        {"parent_block_id": "concept1", "concept_name": "Calculus", "doc_id": "doc1"}
    ]
    selector = QuestionSelector(profile_manager=mock_profile_manager, retriever=mock_retriever, question_generator=mock_qg_rag)
    
    mock_retriever.get_all_chunks_metadata.assert_called_once()
    assert len(selector.curriculum_map) == 1
    assert selector.curriculum_map[0]['concept_id'] == "concept1"

@pytest.mark.asyncio
async def test_determine_question_params(question_selector):
    """Test the question parameter determination logic."""
    # Test for new concept (no knowledge)
    params = await question_selector._determine_question_params("learner1", None)
    assert params["difficulty"] == "beginner"
    assert params["type"] == "factual"
    
    # Test for concept with low score
    mock_knowledge = {
        "current_score": 4.0,
        "total_attempts": 2,
        "current_difficulty_level": "beginner"
    }
    question_selector.profile_manager.get_concept_knowledge.return_value = mock_knowledge
    params = await question_selector._determine_question_params("learner1", "concept1")
    assert params["difficulty"] == "beginner"
    assert params["type"] in ["factual", "conceptual"]

@pytest.mark.asyncio
async def test_select_concept_for_review(question_selector, mock_profile_manager):
    """Test selecting a concept for review."""
    mock_profile_manager.get_concepts_for_review.return_value = [
        {
            "concept_id": "concept1",
            "current_score": 7.0,
            "next_review_at": "2024-03-20"
        }
    ]
    
    question_selector.curriculum_map = [
        {
            "concept_id": "concept1",
            "concept_name": "Calculus",
            "doc_id": "doc1"
        }
    ]
    
    selected = await question_selector._select_concept_for_review("learner1")
    assert selected is not None
    assert selected["concept_id"] == "concept1"
    assert selected["is_review"] is True

@pytest.mark.asyncio
async def test_select_new_concept(question_selector, mock_profile_manager):
    """Test selecting a new concept."""
    question_selector.curriculum_map = [
        {
            "concept_id": "concept1",
            "concept_name": "Calculus",
            "doc_id": "doc1"
        },
        {
            "concept_id": "concept2",
            "concept_name": "Algebra",
            "doc_id": "doc2"
        }
    ]
    
    mock_profile_manager.get_concept_knowledge.return_value = None
    
    selected = await question_selector._select_new_concept("learner1")
    assert selected is not None
    assert selected["concept_id"] in ["concept1", "concept2"]
    assert selected["is_review"] is False

@pytest.mark.asyncio
async def test_select_next_question(question_selector, mock_profile_manager, mock_retriever):
    """Test the main question selection flow."""
    # Mock a concept for review
    mock_profile_manager.get_concepts_for_review.return_value = [
        {
            "concept_id": "concept1",
            "current_score": 7.0,
            "next_review_at": "2024-03-20"
        }
    ]
    
    question_selector.curriculum_map = [
        {
            "concept_id": "concept1",
            "concept_name": "Calculus",
            "doc_id": "doc1"
        }
    ]
    
    mock_retriever.get_chunks_for_parent_block.return_value = [
        {"chunk_text": "This is a test chunk."}
    ]
    
    question_selector.question_generator.generate_questions.return_value = [
        {"question": "What is calculus?", "answer": "A branch of mathematics."}
    ]
    
    result = await question_selector.select_next_question("learner1")
    assert result is not None
    assert "error" not in result