import pytest
from unittest.mock import MagicMock, AsyncMock
from src.learner_model.knowledge_tracker import KnowledgeTracker
from src.learner_model.profile_manager import LearnerProfileManager
from src.adaptive_engine.srs_scheduler import SRSScheduler

@pytest.fixture
def mock_profile_manager():
    """Fixture for a mocked LearnerProfileManager."""
    return MagicMock(spec=LearnerProfileManager)

@pytest.fixture
def mock_srs_scheduler():
    """Fixture for a mocked SRSScheduler."""
    return MagicMock(spec=SRSScheduler)

@pytest.fixture
def knowledge_tracker(mock_profile_manager, mock_srs_scheduler):
    """Create a KnowledgeTracker instance with mocked dependencies."""
    return KnowledgeTracker(profile_manager=mock_profile_manager, srs_scheduler=mock_srs_scheduler)

def test_calculate_graded_score(knowledge_tracker):
    """Test the graded score calculation."""
    score = knowledge_tracker._calculate_graded_score(0.8)
    assert score == 8.0

    score = knowledge_tracker._calculate_graded_score(0.4)
    assert score == 4.0

    score = knowledge_tracker._calculate_graded_score(0.3)
    assert score == 3.0

@pytest.mark.asyncio
async def test_update_knowledge_level_successful(knowledge_tracker, mock_profile_manager, mock_srs_scheduler):
    """Test updating knowledge level for a successful answer."""
    learner_id = "learner1"
    concept_id = "concept1"
    doc_id = "doc1"
    accuracy_score = 0.9
    raw_evaluation = {"similarity_score": 0.9}
    
    mock_srs_scheduler.calculate_next_review_details.return_value = {"new": "srs_details"}
    mock_profile_manager.get_concept_knowledge.return_value = None

    await knowledge_tracker.update_knowledge_level(learner_id, concept_id, doc_id, accuracy_score, raw_evaluation)

    final_score = knowledge_tracker._calculate_graded_score(accuracy_score)
    
    mock_profile_manager.update_concept_srs_and_difficulty.assert_called_once_with(
        learner_id=learner_id, 
        concept_id=concept_id,
        doc_id=doc_id,
        score=final_score, 
        answered_correctly=True, 
        srs_details={"new": "srs_details"},
        raw_eval_data=raw_evaluation
    )

@pytest.mark.asyncio
async def test_update_knowledge_level_incorrect_answer(knowledge_tracker, mock_profile_manager, mock_srs_scheduler):
    """Test updating knowledge level for an incorrect answer."""
    learner_id = "learner2"
    concept_id = "concept2"
    doc_id = "doc2"
    accuracy_score = 0.4
    raw_evaluation = {"similarity_score": 0.4}

    mock_srs_scheduler.calculate_next_review_details.return_value = {"new": "srs_details_incorrect"}
    mock_profile_manager.get_concept_knowledge.return_value = None

    await knowledge_tracker.update_knowledge_level(learner_id, concept_id, doc_id, accuracy_score, raw_evaluation)

    final_score = knowledge_tracker._calculate_graded_score(accuracy_score)
    
    mock_profile_manager.update_concept_srs_and_difficulty.assert_called_once_with(
        learner_id=learner_id, 
        concept_id=concept_id,
        doc_id=doc_id,
        score=final_score, 
        answered_correctly=False, 
        srs_details={"new": "srs_details_incorrect"},
        raw_eval_data=raw_evaluation
    )

@pytest.mark.asyncio
async def test_update_knowledge_level_profile_manager_fails(knowledge_tracker, mock_profile_manager, mock_srs_scheduler):
    """Test that an exception from the profile manager is propagated."""
    learner_id = "learner3"
    concept_id = "concept3"
    doc_id = "doc3"
    accuracy_score = 0.95
    raw_evaluation = {"similarity_score": 0.95}

    mock_srs_scheduler.calculate_next_review_details.return_value = {"new": "srs_details"}
    mock_profile_manager.update_concept_srs_and_difficulty.side_effect = Exception("DB error")
    mock_profile_manager.get_concept_knowledge.return_value = None

    with pytest.raises(Exception, match="DB error"):
        await knowledge_tracker.update_knowledge_level(learner_id, concept_id, doc_id, accuracy_score, raw_evaluation)