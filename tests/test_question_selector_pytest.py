import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.adaptive_engine.question_selector import QuestionSelector
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.question_generator_rag import RAGQuestionGenerator
from src.learner_model.profile_manager import LearnerProfileManager

# --- Fixtures ---

@pytest.fixture
def mock_retriever():
    """Fixture for a mocked HybridRetriever."""
    retriever = MagicMock(spec=HybridRetriever)
    retriever.get_all_chunks_metadata = AsyncMock(return_value=[])
    retriever.get_chunks_for_parent_block = AsyncMock(return_value=[])
    return retriever

@pytest.fixture
def mock_profile_manager():
    """Fixture for a mocked LearnerProfileManager."""
    pm = MagicMock(spec=LearnerProfileManager)
    pm.get_concepts_for_review.return_value = []
    pm.get_concept_knowledge.return_value = None
    return pm

@pytest.fixture
def mock_question_generator():
    """Fixture for a mocked RAGQuestionGenerator."""
    generator = MagicMock(spec=RAGQuestionGenerator)
    generator.generate_question = AsyncMock(return_value=(
        "Test question",
        "Test answer",
        "Test explanation"
    ))
    return generator

@pytest.fixture
def question_selector(mock_retriever, mock_question_generator):
    """Fixture for a QuestionSelector instance with mocked dependencies."""
    profile_manager = MagicMock(spec=LearnerProfileManager)
    return QuestionSelector(
        profile_manager=profile_manager,
        retriever=mock_retriever,
        question_generator=mock_question_generator
    )

@pytest.fixture
async def question_selector():
    """Async fixture to create and initialize a QuestionSelector instance."""
    mock_retriever = AsyncMock(spec=HybridRetriever)
    mock_question_generator = AsyncMock(spec=RAGQuestionGenerator)
    mock_profile_manager = AsyncMock(spec=LearnerProfileManager)

    # Configure mock return values
    mock_retriever.get_all_chunks_metadata.return_value = [
        {"chunk_id": "concept1", "text": "Calculus concept", "doc_id": "doc1"},
        {"chunk_id": "concept2", "text": "Algebra concept", "doc_id": "doc2"}
    ]
    mock_retriever.get_chunks_for_parent_block.return_value = [
        {"chunk_id": "1", "chunk_text": "content"}
    ]
    mock_profile_manager.get_concepts_for_review.return_value = []
    mock_profile_manager.get_concept_knowledge.return_value = None
    mock_question_generator.generate_question.return_value = (
        "Test question",
        "Test answer",
        "Test explanation"
    )

    selector = QuestionSelector(
        profile_manager=mock_profile_manager,
        retriever=mock_retriever,
        question_generator=mock_question_generator
    )
    await selector.initialize()  # Initialize the curriculum map

    # Attach mocks for easy access in tests
    selector.mock_retriever = mock_retriever
    selector.mock_profile_manager = mock_profile_manager
    selector.mock_question_generator = mock_question_generator
    
    return selector

# --- Test Cases ---

@pytest.mark.asyncio
async def test_initialization_and_curriculum_map_loading(question_selector):
    """Test that QuestionSelector initializes and loads the curriculum map."""
    question_selector.mock_retriever.get_all_chunks_metadata.assert_awaited_once()
    assert len(question_selector.curriculum_map) == 2
    assert question_selector.curriculum_map[0]['concept_id'] == "concept1"
    assert question_selector.curriculum_map[1]['concept_id'] == "concept2"

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
    question_selector.mock_profile_manager.get_concept_knowledge.return_value = mock_knowledge
    params = await question_selector._determine_question_params("learner1", "1")
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
async def test_select_next_question_no_concepts(question_selector):
    """Test question selection when there are no concepts for review."""
    question_selector.mock_retriever.get_all_chunks_metadata.return_value = []
    
    result = await question_selector.select_next_question("learner1")
    
    assert result is None
    question_selector.mock_retriever.get_all_chunks_metadata.assert_awaited_once()

@pytest.mark.asyncio
async def test_select_next_question_review_concepts(question_selector):
    """Test question selection when there are concepts for review."""
    question_selector.mock_profile_manager.get_concept_knowledge.return_value = {
        "current_score": 0.2  # Below review threshold
    }
    
    result = await question_selector.select_next_question("learner1")
    
    assert result is not None
    assert result[0] == "Test question"
    assert result[1] == "Test answer"
    assert result[2] == "Test explanation"
    question_selector.mock_retriever.get_all_chunks_metadata.assert_awaited_once()
    question_selector.mock_question_generator.generate_question.assert_awaited_once()

@pytest.mark.asyncio
async def test_select_next_question_new_concepts(question_selector):
    """Test question selection when there are new concepts to learn."""
    question_selector.mock_profile_manager.get_concept_knowledge.return_value = None  # New concept
    
    result = await question_selector.select_next_question("learner1")
    
    assert result is not None
    assert result[0] == "Test question"
    assert result[1] == "Test answer"
    assert result[2] == "Test explanation"
    question_selector.mock_retriever.get_all_chunks_metadata.assert_awaited_once()
    question_selector.mock_question_generator.generate_question.assert_awaited_once()