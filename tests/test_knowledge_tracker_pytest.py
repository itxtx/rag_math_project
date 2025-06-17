import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

from src.learner_model.knowledge_tracker import KnowledgeTracker
from src.learner_model.profile_manager import LearnerProfileManager

@pytest.fixture
def mock_profile_manager():
    """Create a mock profile manager."""
    manager = MagicMock(spec=LearnerProfileManager)
    manager.update_concept_srs_and_difficulty.return_value = True
    return manager

@pytest.fixture
def knowledge_tracker(mock_profile_manager):
    """Create a KnowledgeTracker instance with mocked dependencies."""
    return KnowledgeTracker(profile_manager=mock_profile_manager)

@pytest.fixture
def test_ids():
    """Provide test IDs for learner and concept."""
    return {
        "learner_id": "learner_kt_001",
        "concept_id": "concept_kt_alpha"
    }

def test_calculate_graded_score(knowledge_tracker):
    """Test the graded score calculation logic."""
    assert knowledge_tracker._calculate_graded_score(1.0) == 10.0
    assert knowledge_tracker._calculate_graded_score(0.0) == 0.0
    assert knowledge_tracker._calculate_graded_score(0.75) == 7.5
    assert knowledge_tracker._calculate_graded_score(0.66666) == 6.67
    assert knowledge_tracker._calculate_graded_score(1.5) == 10.0
    assert knowledge_tracker._calculate_graded_score(-0.5) == 0.0

@pytest.mark.asyncio
async def test_update_knowledge_level_successful(knowledge_tracker, mock_profile_manager, test_ids):
    """Test successful knowledge level update."""
    accuracy = 0.85
    expected_graded_score = 8.5
    expected_answered_correctly = True
    raw_eval_data = {"feedback": "Very good understanding."}

    success = await knowledge_tracker.update_knowledge_level(
        test_ids["learner_id"],
        test_ids["concept_id"],
        accuracy,
        raw_eval_data
    )

    assert success is True
    mock_profile_manager.update_concept_srs_and_difficulty.assert_called_once_with(
        learner_id=test_ids["learner_id"],
        concept_id=test_ids["concept_id"],
        score=expected_graded_score,
        answered_correctly=expected_answered_correctly,
        raw_eval_data=raw_eval_data
    )

@pytest.mark.asyncio
async def test_update_knowledge_level_incorrect_answer(knowledge_tracker, mock_profile_manager, test_ids):
    """Test knowledge level update with incorrect answer."""
    accuracy = 0.4
    expected_graded_score = 4.0
    expected_answered_correctly = False
    raw_eval_data = {"feedback": "Needs more work."}

    success = await knowledge_tracker.update_knowledge_level(
        test_ids["learner_id"],
        test_ids["concept_id"],
        accuracy,
        raw_eval_data
    )

    assert success is True
    mock_profile_manager.update_concept_srs_and_difficulty.assert_called_once_with(
        learner_id=test_ids["learner_id"],
        concept_id=test_ids["concept_id"],
        score=expected_graded_score,
        answered_correctly=expected_answered_correctly,
        raw_eval_data=raw_eval_data
    )

@pytest.mark.asyncio
async def test_update_knowledge_level_profile_manager_fails(knowledge_tracker, mock_profile_manager, test_ids):
    """Test knowledge level update when profile manager fails."""
    mock_profile_manager.update_concept_srs_and_difficulty.return_value = False
    
    accuracy = 0.9
    raw_eval_data = {"feedback": "Excellent."}

    success = await knowledge_tracker.update_knowledge_level(
        test_ids["learner_id"],
        test_ids["concept_id"],
        accuracy,
        raw_eval_data
    )

    assert success is False, "Update should fail if profile manager reports failure"
    mock_profile_manager.update_concept_srs_and_difficulty.assert_called_once()

def test_init_with_invalid_profile_manager():
    """Test initialization with invalid profile manager."""
    with pytest.raises(TypeError):
        KnowledgeTracker(profile_manager="not_a_profile_manager_instance") 