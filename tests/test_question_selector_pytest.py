import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
import os
import numpy as np

from src.adaptive_engine.question_selector import QuestionSelector, LOW_SCORE_THRESHOLD, PERFECT_SCORE_THRESHOLD
from src.learner_model.profile_manager import LearnerProfileManager
from src.retrieval.retriever import HybridRetriever
from src.generation.question_generator_rag import RAGQuestionGenerator
from src import config
from sentence_transformers import SentenceTransformer

# Test data
SAMPLE_CHUNK_META = [
    {"parent_block_id": "pb_intro", "concept_name": "Introduction", "concept_type": "section", "source_path": "doc1.tex", "doc_id": "doc1"},
    {"parent_block_id": "pb_intro_sub1", "concept_name": "Background", "concept_type": "subsection", "source_path": "doc1.tex", "doc_id": "doc1"},
    {"parent_block_id": "pb_vectors", "concept_name": "Vectors", "concept_type": "section", "source_path": "doc1.tex", "doc_id": "doc1"},
    {"parent_block_id": "pb_matrices", "concept_name": "Matrices", "concept_type": "section", "source_path": "doc2.tex", "doc_id": "doc2"},
]

@pytest.fixture
def mock_profile_manager():
    return MagicMock(spec=LearnerProfileManager)

@pytest.fixture
def mock_retriever():
    retriever = MagicMock(spec=HybridRetriever)
    retriever.get_all_chunks_metadata.return_value = list(SAMPLE_CHUNK_META)
    return retriever

@pytest.fixture
def mock_question_generator():
    return MagicMock(spec=RAGQuestionGenerator)

@pytest.fixture
def question_selector(mock_profile_manager, mock_retriever, mock_question_generator):
    return QuestionSelector(
        profile_manager=mock_profile_manager,
        retriever=mock_retriever,
        question_generator=mock_question_generator
    )

@pytest.mark.asyncio
async def test_initialization_and_curriculum_map_loading(question_selector, mock_retriever):
    """Test that the selector initializes correctly and loads the curriculum map."""
    mock_retriever.get_all_chunks_metadata.assert_called_once()
    assert len(question_selector.curriculum_map) == 4
    
    intro_concept = next((c for c in question_selector.curriculum_map if c["concept_id"] == "pb_intro"), None)
    assert intro_concept is not None
    assert intro_concept["concept_name"] == "Introduction"

@pytest.mark.asyncio
async def test_load_curriculum_map_empty_metadata(mock_profile_manager, mock_retriever, mock_question_generator):
    """Test handling of empty metadata when loading curriculum map."""
    mock_retriever.get_all_chunks_metadata.return_value = []
    selector = QuestionSelector(
        profile_manager=mock_profile_manager,
        retriever=mock_retriever,
        question_generator=mock_question_generator
    )
    assert len(selector.curriculum_map) == 0

@pytest.mark.asyncio
async def test_determine_difficulty(question_selector, mock_profile_manager):
    """Test difficulty determination based on learner's knowledge."""
    learner_id = "learner1"
    concept_id_new = "new_concept"
    concept_id_low_score = "low_score_concept"
    concept_id_mid_score = "mid_score_concept"
    concept_id_high_score = "high_score_concept"

    def get_knowledge_side_effect(lid, cid):
        if cid == concept_id_low_score:
            return {"current_score": LOW_SCORE_THRESHOLD - 1, "total_attempts": 1}
        if cid == concept_id_mid_score:
            return {"current_score": (LOW_SCORE_THRESHOLD + PERFECT_SCORE_THRESHOLD) / 2, "total_attempts": 2}
        if cid == concept_id_high_score:
            return {"current_score": PERFECT_SCORE_THRESHOLD, "total_attempts": 3}
        return None

    mock_profile_manager.get_concept_knowledge.side_effect = get_knowledge_side_effect

    assert await question_selector._determine_difficulty(learner_id, concept_id_new) == "beginner"
    assert await question_selector._determine_difficulty(learner_id, concept_id_low_score) == "beginner"
    assert await question_selector._determine_difficulty(learner_id, concept_id_mid_score) == "intermediate"
    assert await question_selector._determine_difficulty(learner_id, concept_id_high_score) == "advanced"
    assert await question_selector._determine_difficulty(learner_id, None) == "beginner"

@pytest.mark.asyncio
async def test_select_concept_for_review(question_selector, mock_retriever, mock_profile_manager):
    """Test selection of concepts for review based on learner's performance."""
    learner_id = "learner_review"
    
    mock_retriever.search.return_value = [
        {"_id": "pb_intro", "chunk_id": "pb_intro", "concept_name": "Introduction", "chunk_text": "Intro text."},
        {"_id": "pb_vectors", "chunk_id": "pb_vectors", "concept_name": "Vectors", "chunk_text": "Vector text."},
        {"_id": "pb_matrices", "chunk_id": "pb_matrices", "concept_name": "Matrices", "chunk_text": "Matrix text."},
    ]
    
    def get_knowledge_side_effect_review(lid, cid):
        if cid == "pb_vectors":
            return {"current_score": LOW_SCORE_THRESHOLD - 1.0, "total_attempts": 1}
        if cid == "pb_intro":
            return {"current_score": 8.0, "total_attempts": 2}
        if cid == "pb_matrices":
            return {"current_score": PERFECT_SCORE_THRESHOLD, "total_attempts": 3}
        return None

    mock_profile_manager.get_concept_knowledge.side_effect = get_knowledge_side_effect_review

    selected = await question_selector._select_concept_for_review(learner_id)
    assert selected is not None
    assert selected["concept_id"] == "pb_vectors"
    assert selected["is_review"] is True

    # Test case where no review is needed
    mock_profile_manager.get_concept_knowledge.side_effect = lambda lid, cid: {
        "pb_intro": {"current_score": 8.0, "total_attempts": 2},
        "pb_vectors": {"current_score": LOW_SCORE_THRESHOLD + 1.0, "total_attempts": 1},
    }.get(cid)
    
    selected_none = await question_selector._select_concept_for_review(learner_id)
    assert selected_none is None

@pytest.mark.asyncio
async def test_select_new_concept(question_selector, mock_profile_manager):
    """Test selection of new concepts based on learner's progress."""
    learner_id = "learner_new"
    
    def get_knowledge_side_effect_new(lid, cid):
        if cid == "pb_intro":
            return {"current_score": PERFECT_SCORE_THRESHOLD, "total_attempts": 2}
        if cid == "pb_vectors":
            return {"current_score": LOW_SCORE_THRESHOLD + 1.0, "total_attempts": 1}
        return None

    mock_profile_manager.get_concept_knowledge.side_effect = get_knowledge_side_effect_new

    selected = await question_selector._select_new_concept(learner_id)
    assert selected is not None
    assert selected["concept_id"] in ["pb_intro_sub1", "pb_matrices", "pb_vectors"]
    assert selected["is_review"] is False

    # Test with last attempted document
    selected_adj = await question_selector._select_new_concept(learner_id, last_attempted_doc_id="doc1")
    assert selected_adj is not None
    assert selected_adj["concept_id"] in ["pb_intro_sub1", "pb_vectors"]
    assert selected_adj.get("doc_id") == "doc1"

    # Test when all concepts are mastered
    mastered_knowledge = {"current_score": PERFECT_SCORE_THRESHOLD, "total_attempts": 1}
    mock_profile_manager.get_concept_knowledge.return_value = mastered_knowledge
    mock_profile_manager.get_concept_knowledge.side_effect = None
    
    selected_none = await question_selector._select_new_concept(learner_id)
    assert selected_none is None

@pytest.mark.asyncio
async def test_select_next_question_flow_chooses_review(question_selector, mock_retriever, mock_question_generator):
    """Test the complete flow when selecting a question for review."""
    learner_id = "learner_flow_review"
    question_selector.curriculum_map = [{"concept_id": "concept_review", "concept_name": "Review Me", "doc_id": "doc1"}]
    
    mock_review_concept_info = {
        "concept_id": "concept_review",
        "concept_name": "Review Me",
        "current_score": 3.0,
        "is_review": True,
        "doc_id": "doc1"
    }
    
    question_selector._select_concept_for_review = AsyncMock(return_value=mock_review_concept_info)
    question_selector._select_new_concept = AsyncMock(return_value=None)
    mock_retriever.get_chunks_for_parent_block.return_value = [{"chunk_text": "Context for review.", "chunk_id": "chunk_rev1"}]
    mock_question_generator.generate_questions = AsyncMock(return_value=["Review question?"])
    
    result = await question_selector.select_next_question(learner_id)
    assert result is not None
    assert "error" not in result
    assert result["concept_id"] == "concept_review"
    
    args, kwargs = mock_question_generator.generate_questions.call_args
    assert kwargs.get("difficulty_level") == "beginner"

@pytest.mark.asyncio
async def test_select_next_question_flow_chooses_new(question_selector, mock_retriever, mock_question_generator):
    """Test the complete flow when selecting a new question."""
    learner_id = "learner_flow_new"
    question_selector.curriculum_map = [{"concept_id": "concept_new", "concept_name": "New Concept", "doc_id": "doc_new"}]
    
    question_selector._select_concept_for_review = AsyncMock(return_value=None)
    mock_new_concept_info = {
        "concept_id": "concept_new",
        "concept_name": "New Concept",
        "is_review": False,
        "doc_id": "doc_new"
    }
    question_selector._select_new_concept = AsyncMock(return_value=mock_new_concept_info)
    mock_retriever.get_chunks_for_parent_block.return_value = [{"chunk_text": "Context for new concept.", "chunk_id": "chunk_new1"}]
    mock_question_generator.generate_questions = AsyncMock(return_value=["New question?"])
    
    result = await question_selector.select_next_question(learner_id)
    assert result is not None
    assert "error" not in result
    assert result["concept_id"] == "concept_new"
    
    args, kwargs = mock_question_generator.generate_questions.call_args
    assert kwargs.get("difficulty_level") == "beginner"

@pytest.mark.asyncio
async def test_select_next_question_no_concept_found(question_selector):
    """Test handling when no concept is found for review or new questions."""
    learner_id = "learner_flow_none"
    question_selector._select_concept_for_review = AsyncMock(return_value=None)
    question_selector._select_new_concept = AsyncMock(return_value=None)

    result = await question_selector.select_next_question(learner_id)
    assert result is not None
    assert "error" in result

@pytest.mark.asyncio
async def test_select_next_question_no_context_retrieved(question_selector, mock_retriever, mock_profile_manager):
    """Test handling when no context can be retrieved for a concept."""
    learner_id = "learner_flow_no_context"
    mock_concept_info = {
        "concept_id": "concept_no_ctx",
        "concept_name": "No Context Concept",
        "is_review": False,
        "doc_id": "doc_no_ctx"
    }
    
    question_selector._select_concept_for_review = AsyncMock(return_value=None)
    question_selector._select_new_concept = AsyncMock(return_value=mock_concept_info)
    mock_retriever.get_chunks_for_parent_block.return_value = []
    mock_profile_manager.get_concept_knowledge.side_effect = lambda lid, cid: None if cid == "concept_no_ctx" else {"current_score": 0.0, "total_attempts": 0}

    result = await question_selector.select_next_question(learner_id)
    assert result is not None
    assert "error" in result
    mock_retriever.get_chunks_for_parent_block.assert_called_once_with("concept_no_ctx", limit=pytest.ANY)

@pytest.mark.asyncio
async def test_select_next_question_qg_fails(question_selector, mock_retriever, mock_question_generator, mock_profile_manager):
    """Test handling when question generation fails."""
    learner_id = "learner_flow_qg_fail"
    mock_concept_info = {
        "concept_id": "concept_qg_fail",
        "concept_name": "QG Fail Concept",
        "is_review": False,
        "doc_id": "doc_qg_fail"
    }
    
    question_selector._select_concept_for_review = AsyncMock(return_value=None)
    question_selector._select_new_concept = AsyncMock(return_value=mock_concept_info)
    mock_retriever.get_chunks_for_parent_block.return_value = [{"chunk_text": "Some context."}]
    mock_question_generator.generate_questions = AsyncMock(return_value=[])
    mock_profile_manager.get_concept_knowledge.side_effect = lambda lid, cid: None if cid == "concept_qg_fail" else {"current_score": 0.0, "total_attempts": 0}

    result = await question_selector.select_next_question(learner_id)
    assert result is not None
    assert "error" in result
    mock_question_generator.generate_questions.assert_called_once() 