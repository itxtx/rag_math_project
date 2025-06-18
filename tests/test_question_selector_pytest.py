import pytest
from unittest.mock import MagicMock, patch
from src.adaptive_engine.question_selector import QuestionSelector
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.question_generator_rag import RAGQuestionGenerator
from src.learner_model.profile_manager import LearnerProfileManager

# --- Fixtures ---

@pytest.fixture
def mock_retriever():
    """Fixture for a mocked HybridRetriever."""
    retriever = MagicMock(spec=HybridRetriever)
    retriever.retrieve_documents.return_value = [] 
    return retriever

@pytest.fixture
def mock_qg_rag():
    """Fixture for a mocked RAGQuestionGenerator."""
    return MagicMock(spec=RAGQuestionGenerator)

@pytest.fixture
def mock_profile_manager():
    """Fixture for a mocked LearnerProfileManager."""
    return MagicMock(spec=LearnerProfileManager)

@pytest.fixture
def question_selector(mock_retriever, mock_qg_rag, mock_profile_manager):
    """Fixture to create a QuestionSelector instance with mocked dependencies."""
    return QuestionSelector(
        retriever=mock_retriever,
        qg_rag=mock_qg_rag,
        profile_manager=mock_profile_manager
    )

# --- Test Cases ---

def test_initialization_and_curriculum_map_loading(mock_retriever):
    """Test that QuestionSelector initializes and loads the curriculum map."""
    # Setup mock data
    mock_doc = MagicMock()
    mock_doc.metadata = {"concept": "calculus", "difficulty": "medium"}
    mock_doc.page_content = "This is a document about calculus."
    mock_retriever.get_all_documents.return_value = [mock_doc]

    selector = QuestionSelector(retriever=mock_retriever, qg_rag=MagicMock(), profile_manager=MagicMock())
    
    mock_retriever.get_all_documents.assert_called_once()
    assert "calculus" in selector.curriculum_map
    assert selector.curriculum_map["calculus"]["medium"][0]["content"] == "This is a document about calculus."

def test_load_curriculum_map_empty_metadata(mock_retriever):
    """Test loading with documents that have no metadata."""
    mock_doc = MagicMock()
    mock_doc.metadata = {} # Empty metadata
    mock_doc.page_content = "Empty metadata doc."
    mock_retriever.get_all_documents.return_value = [mock_doc]
    
    selector = QuestionSelector(retriever=mock_retriever, qg_rag=MagicMock(), profile_manager=MagicMock())
    
    assert len(selector.curriculum_map) == 0

def test_determine_difficulty(question_selector):
    """Test the difficulty determination logic."""
    difficulty = question_selector._determine_difficulty(0.1)
    assert difficulty == "easy"
    
    difficulty = question_selector._determine_difficulty(0.6)
    assert difficulty == "medium"

    difficulty = question_selector._determine_difficulty(0.9)
    assert difficulty == "hard"

def test_select_concept_for_review(question_selector, mock_profile_manager):
    """Test selecting a concept that is due for review."""
    mock_profile_manager.get_concepts_due_for_review.return_value = [("calculus",)]
    
    concept = question_selector._select_concept_for_review("learner1")
    
    assert concept == "calculus"
    mock_profile_manager.get_concepts_due_for_review.assert_called_once_with("learner1")

def test_select_new_concept(question_selector, mock_profile_manager):
    """Test selecting a new concept for a learner."""
    # Setup
    question_selector.curriculum_map = {"calculus": {}, "algebra": {}}
    mock_profile_manager.get_all_concept_scores.return_value = {"calculus": 0.5}
    
    concept = question_selector._select_new_concept("learner1")
    
    # "algebra" should be chosen as it's unlearned
    assert concept == "algebra"

@patch.object(QuestionSelector, '_select_concept_for_review')
@patch.object(QuestionSelector, '_select_new_concept')
def test_select_next_question_flow_chooses_review(mock_select_new, mock_select_review, question_selector):
    """Test the main flow when a review concept is available."""
    mock_select_review.return_value = "calculus"
    mock_select_new.return_value = "algebra"

    # Mock the retriever to return a chunk for the selected concept
    mock_chunk = {"content": "Review chunk about calculus."}
    question_selector.curriculum_map = {"calculus": {"easy": [mock_chunk]}}
    question_selector.qg_rag.run.return_value = {"question": "What is a derivative?", "answer": "It's a rate of change."}

    result = question_selector.select_next_question("learner1")
    
    mock_select_review.assert_called_once_with("learner1")
    mock_select_new.assert_not_called()
    assert result["question"] == "What is a derivative?"
    assert result["concept"] == "calculus"

@patch.object(QuestionSelector, '_select_concept_for_review')
@patch.object(QuestionSelector, '_select_new_concept')
def test_select_next_question_flow_chooses_new(mock_select_new, mock_select_review, question_selector):
    """Test the main flow when no review concepts are available."""
    mock_select_review.return_value = None
    mock_select_new.return_value = "algebra"
    
    mock_chunk = {"content": "New chunk about algebra."}
    question_selector.curriculum_map = {"algebra": {"medium": [mock_chunk]}}
    question_selector.qg_rag.run.return_value = {"question": "What is a variable?", "answer": "A symbol for a number."}
    
    result = question_selector.select_next_question("learner1")
    
    mock_select_review.assert_called_once_with("learner1")
    mock_select_new.assert_called_once_with("learner1")
    assert result["question"] == "What is a variable?"
    assert result["concept"] == "algebra"

def test_select_next_question_no_concept_found(question_selector):
    """Test flow when no new or review concepts can be found."""
    with patch.object(question_selector, '_select_concept_for_review', return_value=None), \
         patch.object(question_selector, '_select_new_concept', return_value=None):
        
        result = question_selector.select_next_question("learner1")
        assert result is None

def test_select_next_question_no_context_retrieved(question_selector):
    """Test flow when a concept is chosen but no content is available for it."""
    with patch.object(question_selector, '_select_concept_for_review', return_value="calculus"):
        question_selector.curriculum_map = {} # Empty map
        result = question_selector.select_next_question("learner1")
        assert result is None

def test_select_next_question_qg_fails(question_selector):
    """Test flow when the question generator fails."""
    with patch.object(question_selector, '_select_concept_for_review', return_value="calculus"):
        mock_chunk = {"content": "A chunk that causes failure."}
        question_selector.curriculum_map = {"calculus": {"hard": [mock_chunk]}}
        question_selector.qg_rag.run.side_effect = Exception("Generation failed")

        result = question_selector.select_next_question("learner1")
        assert result is None