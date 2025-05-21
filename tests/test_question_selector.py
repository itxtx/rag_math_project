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

    # ... (other tests remain the same) ...

    @async_test
    async def test_select_new_concept(self):
        learner_id = "learner_new"
        
        # Scenario 1: Some new, some unmastered
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

        # Scenario 2: Adjacency
        selected_adj = await self.selector._select_new_concept(learner_id, last_attempted_doc_id="doc1")
        self.assertIsNotNone(selected_adj)
        self.assertIn(selected_adj["concept_id"], ["pb_intro_sub1", "pb_vectors"])
        self.assertEqual(selected_adj.get("doc_id"), "doc1")

        # Scenario 3: All concepts mastered
        print("DEBUG test_select_new_concept: Simulating all concepts mastered.")
        mastered_knowledge = {"current_score": PERFECT_SCORE_THRESHOLD, "total_attempts": 1}
        # Ensure this mock applies to all concept_ids in the curriculum_map
        # by making it the default return value for this part of the test.
        self.mock_profile_manager.get_concept_knowledge.side_effect = None # Clear previous side_effect
        self.mock_profile_manager.get_concept_knowledge.return_value = mastered_knowledge
        
        # Verify the mock for a specific curriculum item
        if self.selector.curriculum_map: # Should not be empty here
            test_cid_from_map = self.selector.curriculum_map[0]['concept_id']
            print(f"DEBUG test_select_new_concept: Knowledge for curriculum item '{test_cid_from_map}' (all mastered scenario): {self.mock_profile_manager.get_concept_knowledge(learner_id, test_cid_from_map)}")
            # This should print the mastered_knowledge dict

        selected_none = await self.selector._select_new_concept(learner_id)
        # If selected_none is not None, print what was selected to debug
        if selected_none is not None:
            print(f"DEBUG test_select_new_concept: 'selected_none' was not None. It was: {selected_none}")
            # Also print knowledge for the selected concept_id
            knowledge_for_selected = self.mock_profile_manager.get_concept_knowledge(learner_id, selected_none['concept_id'])
            print(f"DEBUG test_select_new_concept: Knowledge for selected '{selected_none['concept_id']}': {knowledge_for_selected}")


        self.assertIsNone(selected_none, f"Should return None if all concepts considered mastered. Got: {selected_none}")

    # ... (rest of the tests remain the same) ...
    @async_test
    async def test_select_next_question_no_context_retrieved(self):
        learner_id = "learner_flow_no_context"
        mock_concept_info = {"concept_id": "concept_no_ctx", "concept_name": "No Context Concept", "is_review": False, "doc_id": "doc_no_ctx"}
        self.selector._select_concept_for_review = AsyncMock(return_value=None)
        self.selector._select_new_concept = AsyncMock(return_value=mock_concept_info)
        self.mock_retriever.get_chunks_for_parent_block.return_value = [] 
        # Ensure get_concept_knowledge returns a dict or None for _determine_difficulty
        self.mock_profile_manager.get_concept_knowledge.side_effect = lambda lid, cid: None if cid == "concept_no_ctx" else {"current_score": 0.0, "total_attempts":0} # Fix applied

        result = await self.selector.select_next_question(learner_id)
        self.assertIsNone(result)
        self.mock_retriever.get_chunks_for_parent_block.assert_called_once_with("concept_no_ctx", limit=unittest.mock.ANY)

    @async_test
    async def test_select_next_question_qg_fails(self):
        learner_id = "learner_flow_qg_fail"
        mock_concept_info = {"concept_id": "concept_qg_fail", "concept_name": "QG Fail Concept", "is_review": False, "doc_id":"doc_qg_fail"}
        self.selector._select_concept_for_review = AsyncMock(return_value=None)
        self.selector._select_new_concept = AsyncMock(return_value=mock_concept_info)
        self.mock_retriever.get_chunks_for_parent_block.return_value = [{"chunk_text": "Some context."}]
        self.mock_question_generator.generate_questions = AsyncMock(return_value=[]) 
        self.mock_profile_manager.get_concept_knowledge.side_effect = lambda lid, cid: None if cid == "concept_qg_fail" else {"current_score": 0.0, "total_attempts":0} # Fix applied

        result = await self.selector.select_next_question(learner_id)
        self.assertIsNone(result)
        self.mock_question_generator.generate_questions.assert_called_once()

if __name__ == '__main__':
    print("Run tests using `python -m unittest discover tests` or `python -m unittest tests.adaptive_engine.test_question_selector`")
