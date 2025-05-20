# tests/learner_model/test_knowledge_tracker.py
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

from src.learner_model.knowledge_tracker import KnowledgeTracker
from src.learner_model.profile_manager import LearnerProfileManager # For type hinting and mock spec

class TestKnowledgeTracker(unittest.TestCase):

    def setUp(self):
        # Mock the LearnerProfileManager
        self.mock_profile_manager = MagicMock(spec=LearnerProfileManager)
        # Configure the return value for update_concept_knowledge to be True for successful calls
        self.mock_profile_manager.update_concept_knowledge.return_value = True
        
        self.tracker = KnowledgeTracker(profile_manager=self.mock_profile_manager)
        
        self.learner_id = "learner_kt_001"
        self.concept_id = "concept_kt_alpha"

    def test_calculate_graded_score(self):
        """Test the conversion from accuracy score (0-1) to graded score (0-10)."""
        self.assertEqual(self.tracker._calculate_graded_score(1.0), 10.0)
        self.assertEqual(self.tracker._calculate_graded_score(0.0), 0.0)
        self.assertEqual(self.tracker._calculate_graded_score(0.75), 7.5)
        self.assertEqual(self.tracker._calculate_graded_score(0.66666), 6.67) # Check rounding
        # Test clamping
        self.assertEqual(self.tracker._calculate_graded_score(1.5), 10.0)
        self.assertEqual(self.tracker._calculate_graded_score(-0.5), 0.0)

    async def test_update_knowledge_level_successful(self):
        """Test successful update of knowledge level."""
        accuracy = 0.85
        expected_graded_score = 8.5
        expected_answered_correctly = True # 0.85 >= 0.7 threshold
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

    async def test_update_knowledge_level_incorrect_answer(self):
        """Test update when answer is considered incorrect based on threshold."""
        accuracy = 0.4
        expected_graded_score = 4.0
        expected_answered_correctly = False # 0.4 < 0.7 threshold
        raw_eval_data = {"feedback": "Needs more work."}

        success = await self.tracker.update_knowledge_level(
            self.learner_id, self.concept_id, accuracy, raw_eval_data
        )
        self.assertTrue(success) # Assuming DB update itself is successful
        self.mock_profile_manager.update_concept_knowledge.assert_called_once_with(
            learner_id=self.learner_id,
            concept_id=self.concept_id,
            score=expected_graded_score,
            answered_correctly=expected_answered_correctly,
            raw_eval_data=raw_eval_data
        )

    async def test_update_knowledge_level_profile_manager_fails(self):
        """Test scenario where profile manager fails to update."""
        self.mock_profile_manager.update_concept_knowledge.return_value = False # Simulate failure
        
        accuracy = 0.9
        raw_eval_data = {"feedback": "Excellent."}

        success = await self.tracker.update_knowledge_level(
            self.learner_id, self.concept_id, accuracy, raw_eval_data
        )
        self.assertFalse(success, "Update should fail if profile manager reports failure.")
        self.mock_profile_manager.update_concept_knowledge.assert_called_once()


    def test_init_with_invalid_profile_manager(self):
        """Test that KnowledgeTracker raises TypeError with invalid profile_manager."""
        with self.assertRaises(TypeError):
            KnowledgeTracker(profile_manager="not_a_profile_manager_instance")

# For running async tests from this file directly (less common for full suites)
if __name__ == '__main__':
    # This is a simplified way to run async tests if the file is executed directly.
    # For a full test suite, `python -m unittest discover` is preferred.
    async def run_tests():
        suite = unittest.TestSuite()
        loader = unittest.TestLoader()
        # Add tests from this class
        # Note: unittest TestLoader might not directly support loading async test methods
        # without a specialized runner or adaptation.
        # The standard `python -m unittest` command handles this better.
        # For direct execution, one might run specific async methods.
        
        # Example of running one async test method directly:
        test_instance = TestKnowledgeTracker()
        test_instance.setUp() # Call setUp manually
        await test_instance.test_update_knowledge_level_successful()
        print("One async test (test_update_knowledge_level_successful) executed directly.")

    # asyncio.run(run_tests()) # This setup for direct run is complex with unittest
    print("Please run these tests using `python -m unittest discover tests` or `python -m unittest tests.learner_model.test_knowledge_tracker`")

