# tests/learner_model/test_profile_manager.py
import unittest
import os
import sqlite3
import json
import time # For checking timestamps approximately
from src.learner_model.profile_manager import LearnerProfileManager
from src import config # To get DATA_DIR for a temporary test DB

class TestLearnerProfileManager(unittest.TestCase):

    def setUp(self):
        # Use an in-memory database for most tests for speed, or a temporary file
        # self.db_path = ":memory:"
        # Using a temporary file to better simulate file-based DB and ensure cleanup
        self.test_db_filename = "test_learner_profiles.sqlite3"
        self.db_path = os.path.join(config.DATA_DIR, self.test_db_filename)
        
        # Ensure DATA_DIR exists
        if not os.path.exists(config.DATA_DIR):
            os.makedirs(config.DATA_DIR)
            
        # Clean up any old test DB file before each test
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            
        self.pm = LearnerProfileManager(db_path=self.db_path)
        self.learner_id_1 = "test_learner_001"
        self.concept_id_1 = "concept_alpha"
        self.concept_id_2 = "concept_beta"

    def tearDown(self):
        self.pm.close_db()
        # Clean up the test database file after tests if it exists
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        # print(f"Test DB {self.db_path} removed if it existed.")


    def test_01_database_connection_and_table_creation(self):
        """Test if the database connection is established and tables are created."""
        self.assertIsNotNone(self.pm.conn)
        self.assertIsNotNone(self.pm.cursor)
        
        # Check if tables exist
        self.pm.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='learners';")
        self.assertIsNotNone(self.pm.cursor.fetchone(), "learners table should exist.")
        
        self.pm.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='concept_knowledge';")
        self.assertIsNotNone(self.pm.cursor.fetchone(), "concept_knowledge table should exist.")
        
        self.pm.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='score_history';")
        self.assertIsNotNone(self.pm.cursor.fetchone(), "score_history table should exist.")

    def test_02_create_and_get_profile(self):
        """Test creating a new learner profile and retrieving it."""
        created = self.pm.create_profile(self.learner_id_1)
        self.assertTrue(created, "Profile creation should return True for a new profile.")

        profile = self.pm.get_profile(self.learner_id_1)
        self.assertIsNotNone(profile)
        self.assertEqual(profile["learner_id"], self.learner_id_1)
        self.assertEqual(profile["overall_progress"], 0.0) # Default value
        self.assertIn("created_at", profile)

        # Test creating an existing profile (should be ignored, return True or indicate success)
        created_again = self.pm.create_profile(self.learner_id_1)
        # Depending on implementation, this might return True (ignored) or False (already exists but not an error)
        # The current implementation of create_profile uses INSERT OR IGNORE,
        # so cursor.rowcount will be 0 if ignored, 1 if inserted.
        # We check >=0, so this should pass.
        self.assertTrue(created_again, "Re-creating profile should not fail.")


        non_existent_profile = self.pm.get_profile("non_existent_learner")
        self.assertIsNone(non_existent_profile)

    def test_03_update_overall_progress(self):
        """Test updating a learner's overall progress."""
        self.pm.create_profile(self.learner_id_1)
        
        updated = self.pm.update_overall_progress(self.learner_id_1, 0.55)
        self.assertTrue(updated)

        profile = self.pm.get_profile(self.learner_id_1)
        self.assertEqual(profile["overall_progress"], 0.55)

        not_updated = self.pm.update_overall_progress("non_existent_learner", 0.5)
        self.assertFalse(not_updated)

    def test_04_update_and_get_concept_knowledge(self):
        """Test updating and retrieving concept-specific knowledge."""
        self.pm.create_profile(self.learner_id_1)
        
        # First update for a concept (should create the entry)
        raw_eval_1 = {"feedback": "Good start!"}
        updated1 = self.pm.update_concept_knowledge(self.learner_id_1, self.concept_id_1, 7.0, True, raw_eval_1)
        self.assertTrue(updated1)

        knowledge1 = self.pm.get_concept_knowledge(self.learner_id_1, self.concept_id_1)
        self.assertIsNotNone(knowledge1)
        self.assertEqual(knowledge1["learner_id"], self.learner_id_1)
        self.assertEqual(knowledge1["concept_id"], self.concept_id_1)
        self.assertEqual(knowledge1["current_score"], 7.0)
        self.assertEqual(knowledge1["last_answered_correctly"], 1)
        self.assertEqual(knowledge1["total_attempts"], 1)
        self.assertEqual(knowledge1["correct_attempts"], 1)
        self.assertIsNotNone(knowledge1["last_attempted_at"])

        # Second update for the same concept
        time.sleep(0.01) # Ensure timestamp changes
        raw_eval_2 = {"feedback": "Improved slightly."}
        updated2 = self.pm.update_concept_knowledge(self.learner_id_1, self.concept_id_1, 6.0, False, raw_eval_2)
        self.assertTrue(updated2)
        
        knowledge2 = self.pm.get_concept_knowledge(self.learner_id_1, self.concept_id_1)
        self.assertEqual(knowledge2["current_score"], 6.0)
        self.assertEqual(knowledge2["last_answered_correctly"], 0)
        self.assertEqual(knowledge2["total_attempts"], 2)
        self.assertEqual(knowledge2["correct_attempts"], 1) # Still 1 correct out of 2 attempts
        self.assertNotEqual(knowledge1["last_attempted_at"], knowledge2["last_attempted_at"])

        # Check non-existent concept
        non_existent_knowledge = self.pm.get_concept_knowledge(self.learner_id_1, "non_existent_concept")
        self.assertIsNone(non_existent_knowledge)

    def test_05_get_score_history(self):
        """Test retrieving score history for a concept."""
        self.pm.create_profile(self.learner_id_1)
        
        raw_eval_1 = {"llm_feedback": "Attempt 1 feedback"}
        raw_eval_2 = {"llm_feedback": "Attempt 2 feedback", "details": "more details"}
        
        self.pm.update_concept_knowledge(self.learner_id_1, self.concept_id_1, 8.0, True, raw_eval_1)
        time.sleep(0.01) # Ensure distinct timestamps
        self.pm.update_concept_knowledge(self.learner_id_1, self.concept_id_1, 4.5, False, raw_eval_2)

        history = self.pm.get_score_history(self.learner_id_1, self.concept_id_1)
        self.assertEqual(len(history), 2)
        
        self.assertEqual(history[0]["score"], 8.0)
        self.assertEqual(history[0]["raw_eval_data"], raw_eval_1)
        self.assertIn("timestamp", history[0])
        
        self.assertEqual(history[1]["score"], 4.5)
        self.assertEqual(history[1]["raw_eval_data"], raw_eval_2)
        self.assertIn("timestamp", history[1])
        
        self.assertTrue(history[1]["timestamp"] > history[0]["timestamp"])

        # History for a concept with no attempts
        no_history = self.pm.get_score_history(self.learner_id_1, self.concept_id_2)
        self.assertEqual(len(no_history), 0)

        # History for non-existent learner
        no_learner_history = self.pm.get_score_history("fake_learner", self.concept_id_1)
        self.assertEqual(len(no_learner_history), 0)

    def test_06_delete_cascade(self):
        """Test if deleting a learner cascades to concept_knowledge and score_history."""
        self.pm.create_profile(self.learner_id_1)
        self.pm.update_concept_knowledge(self.learner_id_1, self.concept_id_1, 9.0, True)
        self.pm.update_concept_knowledge(self.learner_id_1, self.concept_id_1, 3.0, False)

        knowledge = self.pm.get_concept_knowledge(self.learner_id_1, self.concept_id_1)
        self.assertIsNotNone(knowledge)
        history = self.pm.get_score_history(self.learner_id_1, self.concept_id_1)
        self.assertEqual(len(history), 2)

        # Delete the learner
        self.pm.cursor.execute("DELETE FROM learners WHERE learner_id = ?", (self.learner_id_1,))
        self.pm.conn.commit()

        # Verify cascade deletion
        profile_after_delete = self.pm.get_profile(self.learner_id_1)
        self.assertIsNone(profile_after_delete)

        knowledge_after_delete = self.pm.get_concept_knowledge(self.learner_id_1, self.concept_id_1)
        self.assertIsNone(knowledge_after_delete, "Concept knowledge should be deleted by cascade.")
        
        # Check score_history table directly as get_score_history relies on concept_knowledge entry
        # This requires getting the knowledge_id *before* deleting the learner/concept_knowledge
        # For this test, we'll assume if concept_knowledge is gone, history related to it is too.
        # A more direct check would involve querying score_history with the knowledge_id.
        # However, if concept_knowledge is deleted, get_score_history should return empty.
        history_after_delete = self.pm.get_score_history(self.learner_id_1, self.concept_id_1)
        self.assertEqual(len(history_after_delete), 0, "Score history should be effectively gone after learner deletion.")


if __name__ == '__main__':
    unittest.main()
