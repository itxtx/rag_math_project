import pytest
from unittest.mock import MagicMock, AsyncMock
from src.adaptive_engine.question_selector import QuestionSelector
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.question_generator_rag import RAGQuestionGenerator
from src.learner_model.profile_manager import LearnerProfileManager

# --- Fixtures ---

@pytest.fixture
def mock_retriever():
    """Fixture for a mocked HybridRetriever."""
    mock = AsyncMock(spec=HybridRetriever)
    # Explicitly add the async methods we need to mock
    mock.get_all_chunks_metadata = AsyncMock(return_value=[])
    mock.get_chunks_for_parent_block = AsyncMock(return_value=[])
    return mock

@pytest.fixture
def mock_question_generator():
    """Fixture for a mocked RAGQuestionGenerator."""
    mock = AsyncMock(spec=RAGQuestionGenerator)
    mock.generate_question = AsyncMock(return_value=(
        "Test question",
        "Test answer",
        "Test explanation"
    ))
    return mock

@pytest.fixture
def mock_profile_manager():
    """Fixture for a mocked LearnerProfileManager."""
    mock = AsyncMock(spec=LearnerProfileManager)
    mock.get_concepts_for_review = AsyncMock(return_value=[])
    mock.get_concept_knowledge = AsyncMock(return_value=None)
    return mock

@pytest.fixture
async def question_selector(mock_profile_manager, mock_retriever, mock_question_generator):
    """Async fixture to create and initialize a QuestionSelector instance."""
    selector = QuestionSelector(
        profile_manager=mock_profile_manager,
        retriever=mock_retriever,
        question_generator=mock_question_generator
    )
    await selector.initialize()
    return selector

# --- Test Cases ---

@pytest.mark.asyncio
async def test_initialization_and_curriculum_map_loading(mock_retriever, mock_question_generator, mock_profile_manager):
    """Test that QuestionSelector initializes and loads the curriculum map."""
    mock_retriever.get_all_chunks_metadata.return_value = [
        {"chunk_id": "concept1", "text": "Calculus concept", "doc_id": "doc1"},
        {"chunk_id": "concept2", "text": "Algebra concept", "doc_id": "doc2"}
    ]
    selector = QuestionSelector(
        profile_manager=mock_profile_manager,
        retriever=mock_retriever,
        question_generator=mock_question_generator
    )
    await selector.initialize()
    
    mock_retriever.get_all_chunks_metadata.assert_awaited_once()
    assert len(selector.curriculum_map) == 2
    assert selector.curriculum_map[0]['concept_id'] == "concept1"
    assert selector.curriculum_map[1]['concept_id'] == "concept2"

@pytest.mark.asyncio
async def test_select_next_question_no_concepts(question_selector, mock_retriever):
    """Test question selection when there are no concepts for review."""
    mock_retriever.get_all_chunks_metadata.return_value = []
    
    result = await question_selector.select_next_question("learner1")
    
    assert result is None
    mock_retriever.get_all_chunks_metadata.assert_awaited_once()

@pytest.mark.asyncio
async def test_select_next_question_review_concepts(question_selector, mock_profile_manager, mock_question_generator):
    """Test question selection when there are concepts for review."""
    mock_profile_manager.get_concept_knowledge.return_value = {
        "current_score": 0.2  # Below review threshold
    }
    
    result = await question_selector.select_next_question("learner1")
    
    assert result is not None
    assert result[0] == "Test question"
    assert result[1] == "Test answer"
    assert result[2] == "Test explanation"
    mock_question_generator.generate_question.assert_awaited_once()

@pytest.mark.asyncio
async def test_select_next_question_new_concepts(question_selector, mock_profile_manager, mock_question_generator):
    """Test question selection when there are new concepts to learn."""
    mock_profile_manager.get_concept_knowledge.return_value = None  # New concept
    
    result = await question_selector.select_next_question("learner1")
    
    assert result is not None
    assert result[0] == "Test question"
    assert result[1] == "Test answer"
    assert result[2] == "Test explanation"
    mock_question_generator.generate_question.assert_awaited_once()