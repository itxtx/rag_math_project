# tests/learner_model/test_knowledge_tracker.py
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

from src.learner_model.knowledge_tracker import KnowledgeTracker
from src.learner_model.profile_manager import LearnerProfileManager 

# Helper to run async test methods
def async_test(coro):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

class TestKnowledgeTracker(unittest.TestCase):

    def setUp(self):
        self.mock_profile_manager = MagicMock(spec=LearnerProfileManager)
        self.mock_profile_manager.update_concept_knowledge.return_value = True 
        
        self.tracker = KnowledgeTracker(profile_manager=self.mock_profile_manager)
        
        self.learner_id = "learner_kt_001"
        self.concept_id = "concept_kt_alpha"

    def test_calculate_graded_score(self):
        self.assertEqual(self.tracker._calculate_graded_score(1.0), 10.0)
        self.assertEqual(self.tracker._calculate_graded_score(0.0), 0.0)
        self.assertEqual(self.tracker._calculate_graded_score(0.75), 7.5)
        self.assertEqual(self.tracker._calculate_graded_score(0.66666), 6.67) 
        self.assertEqual(self.tracker._calculate_graded_score(1.5), 10.0)
        self.assertEqual(self.tracker._calculate_graded_score(-0.5), 0.0)

    @async_test 
    async def test_update_knowledge_level_successful(self):
        accuracy = 0.85
        expected_graded_score = 8.5
        expected_answered_correctly = True 
        raw_eval_data = {"feedback": "Very good understanding."}

        success = await self.tracker.update_knowledge_level(
            self.learner_id, self.concept_id, accuracy, raw_eval_data
        )

        self.assertTrue(success)
        self.mock_profile_manager.update_concept_knowledge.assert_called_once_with(
            learner_id=self.learner_id,
            concept_id=self.concept_id,
            score=expected_graded_score,
            answered_correctly=expected_answered_correctly,
            raw_eval_data=raw_eval_data
        )

    @async_test 
    async def test_update_knowledge_level_incorrect_answer(self):
        accuracy = 0.4
        expected_graded_score = 4.0
        expected_answered_correctly = False 
        raw_eval_data = {"feedback": "Needs more work."}

        success = await self.tracker.update_knowledge_level(
            self.learner_id, self.concept_id, accuracy, raw_eval_data
        )
        self.assertTrue(success) 
        self.mock_profile_manager.update_concept_knowledge.assert_called_once_with(
            learner_id=self.learner_id,
            concept_id=self.concept_id,
            score=expected_graded_score,
            answered_correctly=expected_answered_correctly,
            raw_eval_data=raw_eval_data
        )

    @async_test 
    async def test_update_knowledge_level_profile_manager_fails(self):
        self.mock_profile_manager.update_concept_knowledge.return_value = False 
        
        accuracy = 0.9
        raw_eval_data = {"feedback": "Excellent."}

        success = await self.tracker.update_knowledge_level(
            self.learner_id, self.concept_id, accuracy, raw_eval_data
        )
        self.assertFalse(success, "Update should fail if profile manager reports failure.")
        self.mock_profile_manager.update_concept_knowledge.assert_called_once()


    def test_init_with_invalid_profile_manager(self):
        with self.assertRaises(TypeError):
            KnowledgeTracker(profile_manager="not_a_profile_manager_instance")

if __name__ == '__main__':
    print("Please run these tests using `python -m unittest discover tests` or `python -m unittest tests.learner_model.test_knowledge_tracker`")
