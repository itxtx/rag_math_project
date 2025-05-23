# src/learner_model/knowledge_tracker.py
import datetime
from typing import Dict, Any, Optional

from .profile_manager import LearnerProfileManager, DIFFICULTY_LEVELS, CORRECT_ANSWERS_TO_ADVANCE_DIFFICULTY
from src.adaptive_engine.srs_scheduler import SRSScheduler # Import the SRS scheduler

class KnowledgeTracker:
    """
    Tracks and updates a learner's knowledge level for specific concepts
    based on answer evaluations, incorporating SRS and difficulty progression.
    """

    def __init__(self, profile_manager: LearnerProfileManager, srs_scheduler: SRSScheduler):
        """
        Initializes the KnowledgeTracker.

        Args:
            profile_manager: An instance of LearnerProfileManager to interact with the database.
            srs_scheduler: An instance of SRSScheduler to calculate review schedules.
        """
        if not isinstance(profile_manager, LearnerProfileManager):
            raise TypeError("profile_manager must be an instance of LearnerProfileManager.")
        if not isinstance(srs_scheduler, SRSScheduler):
            raise TypeError("srs_scheduler must be an instance of SRSScheduler.")
            
        self.profile_manager = profile_manager
        self.srs_scheduler = srs_scheduler
        print("KnowledgeTracker initialized with ProfileManager and SRSScheduler.")

    def _calculate_graded_score(self, accuracy_score: float) -> float:
        """
        Converts an accuracy score (0.0 - 1.0) to a graded score (0 - 10).
        """
        accuracy_score = max(0.0, min(1.0, accuracy_score))
        return round(accuracy_score * 10.0, 2)

    async def update_knowledge_level(self,
                                     learner_id: str,
                                     concept_id: str,
                                     accuracy_score: float, 
                                     raw_eval_data: Optional[Dict[str, Any]] = None
                                     ) -> bool:
        """
        Updates the learner's knowledge level for a concept based on the accuracy of an answer.
        This includes updating the score, attempt counts, SRS data, and difficulty level.

        Args:
            learner_id: The ID of the learner.
            concept_id: The unique identifier for the concept/question.
            accuracy_score: The accuracy score from the evaluation (0.0 to 1.0).
            raw_eval_data: Optional dictionary containing the raw evaluation details.

        Returns:
            True if the update was successful, False otherwise.
        """
        print(f"\nKnowledgeTracker: Updating knowledge for learner '{learner_id}', concept '{concept_id}'.")
        print(f"  Received accuracy score: {accuracy_score:.2f}")

        graded_score = self._calculate_graded_score(accuracy_score)
        print(f"  Calculated graded score: {graded_score:.2f}/10.0")

        answered_correctly_threshold = 0.7 
        answered_correctly = accuracy_score >= answered_correctly_threshold
        print(f"  Answered correctly (threshold {answered_correctly_threshold*100}%): {answered_correctly}")

        # Get current SRS state and difficulty for this concept from the profile
        current_knowledge = self.profile_manager.get_concept_knowledge(learner_id, concept_id)
        
        current_srs_repetitions = 0
        current_srs_interval_days = 0
        # current_difficulty_level = DIFFICULTY_LEVELS[0] # Default to 'beginner'
        # consecutive_correct_at_difficulty = 0

        if current_knowledge:
            current_srs_repetitions = current_knowledge.get("srs_repetitions", 0)
            current_srs_interval_days = current_knowledge.get("srs_interval_days", 0)
            # current_difficulty_level = current_knowledge.get("current_difficulty_level", DIFFICULTY_LEVELS[0])
            # consecutive_correct_at_difficulty = current_knowledge.get("consecutive_correct_at_difficulty", 0)
        
        # Calculate next SRS state
        srs_details = self.srs_scheduler.calculate_next_review_details(
            answered_correctly=answered_correctly,
            current_srs_repetitions=current_srs_repetitions,
            current_srs_interval_days=current_srs_interval_days
        )
        # srs_details will contain 'next_interval_days', 'next_review_at', 'new_srs_repetitions'

        # Use ProfileManager to update the database with all new information
        # The update_concept_srs_and_difficulty method handles difficulty progression internally.
        success = self.profile_manager.update_concept_srs_and_difficulty(
            learner_id=learner_id,
            concept_id=concept_id,
            score=graded_score, 
            answered_correctly=answered_correctly,
            srs_details=srs_details, # Pass the dictionary from the scheduler
            raw_eval_data=raw_eval_data
        )

        if success:
            print(f"  Successfully updated knowledge profile (including SRS & difficulty) for concept '{concept_id}'.")
        else:
            print(f"  Failed to update knowledge profile for concept '{concept_id}'.")
        
        return success

async def demo_knowledge_tracker_with_srs():
    print("--- KnowledgeTracker Demo (with SRS) ---")
    
    import os
    from src import config 
    from src.learner_model.profile_manager import LearnerProfileManager # For demo setup

    demo_db_path = os.path.join(config.DATA_DIR, "demo_tracker_srs_profiles.sqlite3")
    if os.path.exists(demo_db_path):
        os.remove(demo_db_path)
        print(f"Removed old demo database: {demo_db_path}")

    profile_manager = LearnerProfileManager(db_path=demo_db_path)
    srs_scheduler = SRSScheduler() # Instantiate the scheduler
    tracker = KnowledgeTracker(profile_manager=profile_manager, srs_scheduler=srs_scheduler)

    learner_id = "learner_srs_demo_001"
    concept_id_A = "concept_A_srs"
    concept_id_B = "concept_B_srs"

    profile_manager.create_profile(learner_id)
    print(f"Profile for {learner_id}: {profile_manager.get_profile(learner_id)}")

    # Simulate interactions for concept_A
    print(f"\n--- Interactions for Concept: {concept_id_A} ---")
    print("\nAttempt 1 (Correct, accuracy 0.9):")
    await tracker.update_knowledge_level(learner_id, concept_id_A, 0.9, {"feedback": "Great start!"})
    k_a1 = profile_manager.get_concept_knowledge(learner_id, concept_id_A)
    print(f"  Knowledge: Score={k_a1.get('current_score')}, Reps={k_a1.get('srs_repetitions')}, Interval={k_a1.get('srs_interval_days')}, NextReview={k_a1.get('next_review_at')}, Difficulty={k_a1.get('current_difficulty_level')}")

    print("\nAttempt 2 (Correct, accuracy 0.95, should advance difficulty if 3 correct at beginner):")
    await tracker.update_knowledge_level(learner_id, concept_id_A, 0.95, {"feedback": "Excellent!"})
    k_a2 = profile_manager.get_concept_knowledge(learner_id, concept_id_A)
    print(f"  Knowledge: Score={k_a2.get('current_score')}, Reps={k_a2.get('srs_repetitions')}, Interval={k_a2.get('srs_interval_days')}, NextReview={k_a2.get('next_review_at')}, Difficulty={k_a2.get('current_difficulty_level')}")

    print("\nAttempt 3 (Correct, accuracy 0.8, should advance difficulty):")
    await tracker.update_knowledge_level(learner_id, concept_id_A, 0.8, {"feedback": "Still good!"})
    k_a3 = profile_manager.get_concept_knowledge(learner_id, concept_id_A)
    print(f"  Knowledge: Score={k_a3.get('current_score')}, Reps={k_a3.get('srs_repetitions')}, Interval={k_a3.get('srs_interval_days')}, NextReview={k_a3.get('next_review_at')}, Difficulty={k_a3.get('current_difficulty_level')}")


    print("\nAttempt 4 (Incorrect, accuracy 0.3):")
    await tracker.update_knowledge_level(learner_id, concept_id_A, 0.3, {"feedback": "Mistake here."})
    k_a4 = profile_manager.get_concept_knowledge(learner_id, concept_id_A)
    print(f"  Knowledge: Score={k_a4.get('current_score')}, Reps={k_a4.get('srs_repetitions')}, Interval={k_a4.get('srs_interval_days')}, NextReview={k_a4.get('next_review_at')}, Difficulty={k_a4.get('current_difficulty_level')}")


    # Simulate interactions for concept_B
    print(f"\n--- Interactions for Concept: {concept_id_B} ---")
    print("\nAttempt 1 (Incorrect, accuracy 0.1):")
    await tracker.update_knowledge_level(learner_id, concept_id_B, 0.1, {"feedback": "Way off."})
    k_b1 = profile_manager.get_concept_knowledge(learner_id, concept_id_B)
    print(f"  Knowledge: Score={k_b1.get('current_score')}, Reps={k_b1.get('srs_repetitions')}, Interval={k_b1.get('srs_interval_days')}, NextReview={k_b1.get('next_review_at')}, Difficulty={k_b1.get('current_difficulty_level')}")


    # Check concepts for review
    print("\nConcepts for review now (should include both if their next_review_at is today):")
    review_concepts = profile_manager.get_concepts_for_review(learner_id)
    for concept in review_concepts:
        print(f"  - ID: {concept['concept_id']}, Next Review: {concept['next_review_at']}, Interval: {concept['srs_interval_days']}, Difficulty: {concept['current_difficulty_level']}")

    profile_manager.close_db()
    print("\n--- KnowledgeTracker SRS Demo Finished ---")
