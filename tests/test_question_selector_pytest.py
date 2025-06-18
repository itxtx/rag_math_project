import pytest
from unittest.mock import AsyncMock
from src.adaptive_engine.question_selector import QuestionSelector
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.question_generator_rag import RAGQuestionGenerator
from src.learner_model.profile_manager import LearnerProfileManager

# --- Fixtures ---

@pytest.fixture
def mock_retriever():
    """Fixture for a mocked HybridRetriever."""
    mock = AsyncMock(spec=HybridRetriever)
    mock.get_all_chunks_metadata = AsyncMock(return_value=[])
    mock.get_chunks_for_parent_block = AsyncMock(return_value=[])
    return mock

@pytest.fixture
def mock_question_generator():
    """Fixture for a mocked RAGQuestionGenerator."""
    mock = AsyncMock(spec=RAGQuestionGenerator)
    mock.generate_questions = AsyncMock(return_value=["Test question"])
    return mock

@pytest.fixture
def mock_profile_manager():
    """Fixture for a mocked LearnerProfileManager."""
    mock = AsyncMock(spec=LearnerProfileManager)
    mock.get_concepts_for_review = AsyncMock(return_value=[])
    mock.get_concept_knowledge = AsyncMock(return_value=None)
    mock.create_profile = AsyncMock(return_value=None)
    # Add missing method
    mock.get_last_attempted_concept_and_doc = AsyncMock(return_value=(None, None))
    return mock

@pytest.fixture
async def question_selector(mock_profile_manager, mock_retriever, mock_question_generator):
    """Async fixture to create and initialize a QuestionSelector instance."""
    selector = QuestionSelector(
        profile_manager=mock_profile_manager,
        retriever=mock_retriever,
        question_generator=mock_question_generator
    )
    # The selector now initializes itself, so we don't need to call it here
    return selector

# --- Test Cases ---

@pytest.mark.asyncio
async def test_initialization_and_curriculum_map_loading(mock_retriever, mock_question_generator, mock_profile_manager):
    """Test that QuestionSelector initializes and loads the curriculum map."""
    mock_retriever.get_all_chunks_metadata.return_value = [
        {"concept_id": "concept1", "text": "Calculus concept", "doc_id": "doc1", "concept_name": "Calculus"},
        {"concept_id": "concept2", "text": "Algebra concept", "doc_id": "doc2", "concept_name": "Algebra"}
    ]
    selector = QuestionSelector(
        profile_manager=mock_profile_manager,
        retriever=mock_retriever,
        question_generator=mock_question_generator
    )
    await selector.initialize()
    
    mock_retriever.get_all_chunks_metadata.assert_awaited_once()
    assert len(selector.curriculum_map) == 2

@pytest.mark.asyncio
async def test_select_next_question_no_concepts(question_selector):
    """Test question selection when there are no concepts available."""
    # Ensure curriculum map is empty for this test
    question_selector.curriculum_map = []
    question_selector.is_initialized.set()

    result = await question_selector.select_next_question("learner1")
    
    assert result['error'] == 'No curriculum content available.'


@pytest.mark.asyncio
async def test_select_next_question_review_concepts(question_selector, mock_profile_manager, mock_question_generator, mock_retriever):
    """Test question selection when there are concepts for review."""
    
    mock_retriever.get_all_chunks_metadata.return_value = [
        {"concept_id": "concept1", "text": "Calculus concept", "doc_id": "doc1", "concept_name": "Calculus"}
    ]
    # We now call initialize() inside the test to ensure the mock is set up first
    await question_selector.initialize()

    mock_profile_manager.get_concepts_for_review.return_value = [
        {"concept_id": "concept1", "current_score": 0.2}
    ]
    mock_retriever.get_chunks_for_parent_block.return_value = [
        {'chunk_text': 'Calculus is the study of change.'}
    ]

    result = await question_selector.select_next_question("learner1")
    
    assert result is not None
    assert result['question_text'] == "Test question"
    mock_question_generator.generate_questions.assert_awaited_once()

@pytest.mark.asyncio
async def test_select_next_question_new_concepts(question_selector, mock_profile_manager, mock_question_generator, mock_retriever):
    """Test question selection when there are new concepts to learn."""
    
    mock_retriever.get_all_chunks_metadata.return_value = [
        {"concept_id": "concept1", "text": "Calculus concept", "doc_id": "doc1", "concept_name": "Calculus"}
    ]
    await question_selector.initialize()

    mock_profile_manager.get_concept_knowledge.return_value = None  # New concept
    mock_retriever.get_chunks_for_parent_block.return_value = [
        {'chunk_text': 'Calculus is the study of change.'}
    ]

    result = await question_selector.select_next_question("learner1")
    
    assert result is not None
    assert result['question_text'] == "Test question"
    mock_question_generator.generate_questions.assert_awaited_once()