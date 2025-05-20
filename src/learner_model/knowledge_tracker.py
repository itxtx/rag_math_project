# src/learner_model/knowledge_tracker.py
import datetime
from typing import Dict, Any, Optional
import asyncio
import os

from .profile_manager import LearnerProfileManager 


class KnowledgeTracker:
    """
    Tracks and updates a learner's knowledge level for specific concepts
    based on answer evaluations.
    """

    def __init__(self, profile_manager: LearnerProfileManager):
        """
        Initializes the KnowledgeTracker.

        Args:
            profile_manager: An instance of LearnerProfileManager to interact with the database.
        """
        if not isinstance(profile_manager, LearnerProfileManager):
            raise TypeError("profile_manager must be an instance of LearnerProfileManager.")
        self.profile_manager = profile_manager
        print("KnowledgeTracker initialized.")

    def _calculate_graded_score(self, accuracy_score: float) -> float:
        """
        Converts an accuracy score (0.0 - 1.0) to a graded score (0 - 10).

        Args:
            accuracy_score: A float between 0.0 and 1.0.

        Returns:
            A float between 0.0 and 10.0.
        """
        # Ensure accuracy_score is within bounds
        accuracy_score = max(0.0, min(1.0, accuracy_score))
        return round(accuracy_score * 10.0, 2) # Rounded to 2 decimal places

    async def update_knowledge_level(self,
                                     learner_id: str,
                                     concept_id: str,
                                     accuracy_score: float, # From AnswerEvaluator (0.0 - 1.0)
                                     raw_eval_data: Optional[Dict[str, Any]] = None
                                     ) -> bool:
        """
        Updates the learner's knowledge level for a concept based on the accuracy of an answer.

        Args:
            learner_id: The ID of the learner.
            concept_id: The unique identifier for the concept/question.
            accuracy_score: The accuracy score from the evaluation (0.0 to 1.0).
            raw_eval_data: Optional dictionary containing the raw evaluation details
                           (e.g., LLM feedback, specific errors) to be logged.

        Returns:
            True if the update was successful, False otherwise.
        """
        print(f"\nKnowledgeTracker: Updating knowledge for learner '{learner_id}', concept '{concept_id}'.")
        print(f"  Received accuracy score: {accuracy_score:.2f}")

        graded_score = self._calculate_graded_score(accuracy_score)
        print(f"  Calculated graded score: {graded_score:.2f}/10.0")

        # Determine if the answer was "correct" based on a threshold.
        # This threshold could be configurable.
        answered_correctly_threshold = 0.7 # e.g., 70% accuracy or higher is "correct"
        answered_correctly = accuracy_score >= answered_correctly_threshold
        print(f"  Answered correctly (threshold {answered_correctly_threshold*100}%): {answered_correctly}")

        # Use ProfileManager to update the database
        success = self.profile_manager.update_concept_knowledge(
            learner_id=learner_id,
            concept_id=concept_id,
            score=graded_score, # Store the 0-10 score
            answered_correctly=answered_correctly,
            raw_eval_data=raw_eval_data
        )

        if success:
            print(f"  Successfully updated knowledge profile for concept '{concept_id}'.")
        else:
            print(f"  Failed to update knowledge profile for concept '{concept_id}'.")
        
        return success

async def demo_knowledge_tracker():
    print("--- KnowledgeTracker Demo ---")
    
    # For this demo, we need a LearnerProfileManager instance.
    # We'll use a temporary in-memory database or a disposable file.
    import os
    from src import config # To get DATA_DIR
    demo_db_path = os.path.join(config.DATA_DIR, "demo_tracker_profiles.sqlite3")
    if os.path.exists(demo_db_path):
        os.remove(demo_db_path)
        print(f"Removed old demo database: {demo_db_path}")

    profile_manager = LearnerProfileManager(db_path=demo_db_path)
    tracker = KnowledgeTracker(profile_manager=profile_manager)

    learner_id = "learner_track_001"
    concept_id_math = "math_concept_algebra"
    concept_id_hist = "hist_concept_ww2"

    # Create profile first
    profile_manager.create_profile(learner_id)
    print(f"Profile for {learner_id}: {profile_manager.get_profile(learner_id)}")


    # Simulate some answer evaluations
    print("\nSimulating first attempt (good answer):")
    eval_data_1 = {"llm_feedback": "Correctly identified all key points.", "raw_score_details": "..."}
    await tracker.update_knowledge_level(learner_id, concept_id_math, 0.95, eval_data_1)

    print("\nSimulating second attempt (poor answer):")
    eval_data_2 = {"llm_feedback": "Missed the core idea.", "errors": ["definition incorrect"]}
    await tracker.update_knowledge_level(learner_id, concept_id_math, 0.20, eval_data_2)
    
    print("\nSimulating third attempt (medium answer):")
    eval_data_3 = {"llm_feedback": "Understood part of it, but some confusion remains."}
    await tracker.update_knowledge_level(learner_id, concept_id_math, 0.65, eval_data_3)

    print("\nSimulating attempt for a different concept:")
    eval_data_4 = {"llm_feedback": "Excellent recall of dates and events."}
    await tracker.update_knowledge_level(learner_id, concept_id_hist, 1.0, eval_data_4)


    # Check the database content via ProfileManager
    print(f"\nFinal knowledge for '{concept_id_math}' for {learner_id}:")
    knowledge_math = profile_manager.get_concept_knowledge(learner_id, concept_id_math)
    if knowledge_math:
        for key, value in knowledge_math.items():
            print(f"  {key}: {value}")
    
    print(f"\nScore history for '{concept_id_math}' for {learner_id}:")
    history_math = profile_manager.get_score_history(learner_id, concept_id_math)
    for entry in history_math:
        print(f"  - Timestamp: {entry['timestamp']}, Score: {entry['score']:.2f}, Eval: {entry.get('raw_eval_data')}")

    print(f"\nFinal knowledge for '{concept_id_hist}' for {learner_id}:")
    knowledge_hist = profile_manager.get_concept_knowledge(learner_id, concept_id_hist)
    if knowledge_hist:
        for key, value in knowledge_hist.items():
            print(f"  {key}: {value}")

    profile_manager.close_db()
    print("\n--- KnowledgeTracker Demo Finished ---")

if __name__ == "__main__":
    # Ensure .env is loaded if config relies on it for DB paths etc.
    # Though for this demo, it's mostly about the DB path in config.DATA_DIR
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        print(f"KnowledgeTracker Demo: Found .env file at {dotenv_path}, loading.")
        from dotenv import load_dotenv
        load_dotenv(dotenv_path)
    
    asyncio.run(demo_knowledge_tracker())
