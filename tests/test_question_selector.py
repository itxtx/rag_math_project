# tests/adaptive_engine/test_question_selector.py
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
import os
import numpy as np 

from src.adaptive_engine.question_selector import QuestionSelector, LOW_SCORE_THRESHOLD, PERFECT_SCORE_THRESHOLD
from src.learner_model.profile_manager import LearnerProfileManager
from src.retrieval.retriever import Retriever
from src.generation.question_generator_rag import RAGQuestionGenerator
from src import config 
from sentence_transformers import SentenceTransformer 

def async_test(coro):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

class TestQuestionSelector(unittest.TestCase):

    def setUp(self):
        self.mock_profile_manager = MagicMock(spec=LearnerProfileManager)
        self.mock_retriever = MagicMock(spec=Retriever)
        self.mock_question_generator = MagicMock(spec=RAGQuestionGenerator)

        self.sample_chunk_meta_for_curriculum = [
            {"parent_block_id": "pb_intro", "concept_name": "Introduction", "concept_type": "section", "source_path": "doc1.tex", "doc_id": "doc1"},
            {"parent_block_id": "pb_intro_sub1", "concept_name": "Background", "concept_type": "subsection", "source_path": "doc1.tex", "doc_id": "doc1"},
            {"parent_block_id": "pb_vectors", "concept_name": "Vectors", "concept_type": "section", "source_path": "doc1.tex", "doc_id": "doc1"},
            {"parent_block_id": "pb_matrices", "concept_name": "Matrices", "concept_type": "section", "source_path": "doc2.tex", "doc_id": "doc2"},
        ]
        self.mock_retriever.get_all_chunks_metadata.return_value = list(self.sample_chunk_meta_for_curriculum)


        self.selector = QuestionSelector(
            profile_manager=self.mock_profile_manager,
            retriever=self.mock_retriever,
            question_generator=self.mock_question_generator
        )
    
    # ... (test_initialization_and_curriculum_map_loading, test_load_curriculum_map_empty_metadata, test_determine_difficulty, test_select_concept_for_review remain the same)
    def test_initialization_and_curriculum_map_loading(self):
        self.mock_retriever.get_all_chunks_metadata.assert_called_once()
        self.assertEqual(len(self.selector.curriculum_map), 4) 
        intro_concept = next((c for c in self.selector.curriculum_map if c["concept_id"] == "pb_intro"), None)
        self.assertIsNotNone(intro_concept)
        self.assertEqual(intro_concept["concept_name"], "Introduction")

    def test_load_curriculum_map_empty_metadata(self):
        self.mock_retriever.get_all_chunks_metadata.return_value = []
        selector = QuestionSelector( 
            profile_manager=self.mock_profile_manager,
            retriever=self.mock_retriever,
            question_generator=self.mock_question_generator
        )
        self.assertEqual(len(selector.curriculum_map), 0)

    @async_test
    async def test_determine_difficulty(self):
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

        self.mock_profile_manager.get_concept_knowledge.side_effect = get_knowledge_side_effect

        self.assertEqual(await self.selector._determine_difficulty(learner_id, concept_id_new), "beginner")
        self.assertEqual(await self.selector._determine_difficulty(learner_id, concept_id_low_score), "beginner")
        self.assertEqual(await self.selector._determine_difficulty(learner_id, concept_id_mid_score), "intermediate")
        self.assertEqual(await self.selector._determine_difficulty(learner_id, concept_id_high_score), "advanced")
        self.assertEqual(await self.selector._determine_difficulty(learner_id, None), "beginner")

    @async_test
    async def test_select_concept_for_review(self):
        learner_id = "learner_review"
        
        self.mock_retriever.search.return_value = [
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
        self.mock_profile_manager.get_concept_knowledge.side_effect = get_knowledge_side_effect_review

        selected = await self.selector._select_concept_for_review(learner_id)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["concept_id"], "pb_vectors")
        self.assertTrue(selected["is_review"])

        self.mock_profile_manager.get_concept_knowledge.side_effect = lambda lid, cid: {
            "pb_intro": {"current_score": 8.0, "total_attempts": 2},
            "pb_vectors": {"current_score": LOW_SCORE_THRESHOLD + 1.0, "total_attempts": 1},
        }.get(cid)
        selected_none = await self.selector._select_concept_for_review(learner_id)
        self.assertIsNone(selected_none)


    @async_test
    async def test_select_new_concept(self):
        learner_id = "learner_new"
        
        def get_knowledge_side_effect_new(lid, cid):
            if cid == "pb_intro": 
                return {"current_score": PERFECT_SCORE_THRESHOLD, "total_attempts": 2} 
            if cid == "pb_vectors": 
                return {"current_score": LOW_SCORE_THRESHOLD + 1.0, "total_attempts": 1} 
            return None 
        self.mock_profile_manager.get_concept_knowledge.side_effect = get_knowledge_side_effect_new

        selected = await self.selector._select_new_concept(learner_id)
        self.assertIsNotNone(selected)
        self.assertIn(selected["concept_id"], ["pb_intro_sub1", "pb_matrices", "pb_vectors"]) 
        self.assertFalse(selected["is_review"])

        selected_adj = await self.selector._select_new_concept(learner_id, last_attempted_doc_id="doc1")
        self.assertIsNotNone(selected_adj)
        self.assertIn(selected_adj["concept_id"], ["pb_intro_sub1", "pb_vectors"])
        self.assertEqual(selected_adj.get("doc_id"), "doc1")

        print("DEBUG test_select_new_concept: Simulating all concepts mastered.")
        mastered_knowledge = {"current_score": PERFECT_SCORE_THRESHOLD, "total_attempts": 1}
        self.mock_profile_manager.get_concept_knowledge.return_value = mastered_knowledge 
        self.mock_profile_manager.get_concept_knowledge.side_effect = None 
        
        selected_none = await self.selector._select_new_concept(learner_id)
        self.assertIsNone(selected_none, f"Should return None if all concepts considered mastered. Got: {selected_none}")


    @async_test
    async def test_select_next_question_flow_chooses_review(self):
        # ... (same as before) ...
        learner_id = "learner_flow_review"
        self.selector.curriculum_map = [{"concept_id": "concept_review", "concept_name": "Review Me", "doc_id": "doc1"}] 
        mock_review_concept_info = {"concept_id": "concept_review", "concept_name": "Review Me", "current_score": 3.0, "is_review": True, "doc_id": "doc1"}
        self.selector._select_concept_for_review = AsyncMock(return_value=mock_review_concept_info)
        self.selector._select_new_concept = AsyncMock(return_value=None) 
        self.mock_retriever.get_chunks_for_parent_block.return_value = [{"chunk_text": "Context for review.", "chunk_id": "chunk_rev1"}]
        self.mock_question_generator.generate_questions = AsyncMock(return_value=["Review question?"])
        self.mock_profile_manager.get_concept_knowledge.return_value = {"current_score": 3.0, "total_attempts":1}
        result = await self.selector.select_next_question(learner_id)
        self.assertIsNotNone(result)
        self.assertNotIn("error", result)
        self.assertEqual(result["concept_id"], "concept_review")
        args, kwargs = self.mock_question_generator.generate_questions.call_args
        self.assertEqual(kwargs.get("difficulty_level"), "beginner")


    @async_test
    async def test_select_next_question_flow_chooses_new(self):
        # ... (same as before) ...
        learner_id = "learner_flow_new"
        self.selector.curriculum_map = [{"concept_id": "concept_new", "concept_name": "New Concept", "doc_id": "doc_new"}]
        self.selector._select_concept_for_review = AsyncMock(return_value=None) 
        mock_new_concept_info = {"concept_id": "concept_new", "concept_name": "New Concept", "is_review": False, "doc_id": "doc_new"}
        self.selector._select_new_concept = AsyncMock(return_value=mock_new_concept_info)
        self.mock_retriever.get_chunks_for_parent_block.return_value = [{"chunk_text": "Context for new concept.", "chunk_id": "chunk_new1"}]
        self.mock_question_generator.generate_questions = AsyncMock(return_value=["New question?"])
        self.mock_profile_manager.get_concept_knowledge.return_value = None 
        result = await self.selector.select_next_question(learner_id)
        self.assertIsNotNone(result)
        self.assertNotIn("error", result)
        self.assertEqual(result["concept_id"], "concept_new")
        args, kwargs = self.mock_question_generator.generate_questions.call_args
        self.assertEqual(kwargs.get("difficulty_level"), "beginner")

    @async_test
    async def test_select_next_question_no_concept_found(self):
        learner_id = "learner_flow_none"
        self.selector._select_concept_for_review = AsyncMock(return_value=None)
        self.selector._select_new_concept = AsyncMock(return_value=None) 

        result = await self.selector.select_next_question(learner_id)
        self.assertIsNotNone(result)
        self.assertIn("error", result) # Expect error dictionary

    @async_test
    async def test_select_next_question_no_context_retrieved(self):
        learner_id = "learner_flow_no_context"
        mock_concept_info = {"concept_id": "concept_no_ctx", "concept_name": "No Context Concept", "is_review": False, "doc_id": "doc_no_ctx"}
        self.selector._select_concept_for_review = AsyncMock(return_value=None)
        self.selector._select_new_concept = AsyncMock(return_value=mock_concept_info)
        self.mock_retriever.get_chunks_for_parent_block.return_value = [] 
        self.mock_profile_manager.get_concept_knowledge.side_effect = lambda lid, cid: None if cid == "concept_no_ctx" else {"current_score": 0.0, "total_attempts":0}

        result = await self.selector.select_next_question(learner_id)
        self.assertIsNotNone(result)
        self.assertIn("error", result) # Expect error dictionary
        self.mock_retriever.get_chunks_for_parent_block.assert_called_once_with("concept_no_ctx", limit=unittest.mock.ANY)


    @async_test
    async def test_select_next_question_qg_fails(self):
        learner_id = "learner_flow_qg_fail"
        mock_concept_info = {"concept_id": "concept_qg_fail", "concept_name": "QG Fail Concept", "is_review": False, "doc_id":"doc_qg_fail"}
        self.selector._select_concept_for_review = AsyncMock(return_value=None)
        self.selector._select_new_concept = AsyncMock(return_value=mock_concept_info)
        self.mock_retriever.get_chunks_for_parent_block.return_value = [{"chunk_text": "Some context."}]
        self.mock_question_generator.generate_questions = AsyncMock(return_value=[]) 
        self.mock_profile_manager.get_concept_knowledge.side_effect = lambda lid, cid: None if cid == "concept_qg_fail" else {"current_score": 0.0, "total_attempts":0}

        result = await self.selector.select_next_question(learner_id)
        self.assertIsNotNone(result)
        self.assertIn("error", result) # Expect error dictionary
        self.mock_question_generator.generate_questions.assert_called_once()


if __name__ == '__main__':
    print("Run tests using `python -m unittest discover tests` or `python -m unittest tests.adaptive_engine.test_question_selector`")
