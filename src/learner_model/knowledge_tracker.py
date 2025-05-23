# src/learner_model/knowledge_tracker.py
import datetime
from typing import Dict, Any, Optional

from .profile_manager import LearnerProfileManager, DIFFICULTY_LEVELS, CORRECT_ANSWERS_TO_ADVANCE_DIFFICULTY
from src.adaptive_engine.srs_scheduler import SRSScheduler 

class KnowledgeTracker:
    """
    Tracks and updates a learner's knowledge level for specific concepts
    based on answer evaluations, incorporating SRS and difficulty progression.
    """

    def __init__(self, profile_manager: LearnerProfileManager, srs_scheduler: SRSScheduler):
        if not isinstance(profile_manager, LearnerProfileManager):
            raise TypeError("profile_manager must be an instance of LearnerProfileManager.")
        if not isinstance(srs_scheduler, SRSScheduler):
            raise TypeError("srs_scheduler must be an instance of SRSScheduler.")
            
        self.profile_manager = profile_manager
        self.srs_scheduler = srs_scheduler
        print("KnowledgeTracker initialized with ProfileManager and SRSScheduler.")

    def _calculate_graded_score(self, accuracy_score: float) -> float:
        accuracy_score = max(0.0, min(1.0, accuracy_score))
        return round(accuracy_score * 10.0, 2)

    async def update_knowledge_level(self,
                                     learner_id: str,
                                     concept_id: str,
                                     doc_id: str, # <<< ADDED doc_id parameter
                                     accuracy_score: float, 
                                     raw_eval_data: Optional[Dict[str, Any]] = None
                                     ) -> bool:
        """
        Updates the learner's knowledge level for a concept.
        """
        print(f"\nKnowledgeTracker: Updating knowledge for learner '{learner_id}', concept '{concept_id}', doc '{doc_id}'.")
        print(f"  Received accuracy score: {accuracy_score:.2f}")

        graded_score = self._calculate_graded_score(accuracy_score)
        print(f"  Calculated graded score: {graded_score:.2f}/10.0")

        answered_correctly_threshold = 0.7 
        answered_correctly = accuracy_score >= answered_correctly_threshold
        print(f"  Answered correctly (threshold {answered_correctly_threshold*100}%): {answered_correctly}")

        current_knowledge = self.profile_manager.get_concept_knowledge(learner_id, concept_id)
        
        current_srs_repetitions = 0
        current_srs_interval_days = 0
        
        if current_knowledge:
            current_srs_repetitions = current_knowledge.get("srs_repetitions", 0)
            current_srs_interval_days = current_knowledge.get("srs_interval_days", 0)
        
        srs_details = self.srs_scheduler.calculate_next_review_details(
            answered_correctly=answered_correctly,
            current_srs_repetitions=current_srs_repetitions,
            current_srs_interval_days=current_srs_interval_days
        )
        
        success = self.profile_manager.update_concept_srs_and_difficulty(
            learner_id=learner_id,
            concept_id=concept_id,
            doc_id=doc_id, # <<< PASS doc_id here
            score=graded_score, 
            answered_correctly=answered_correctly,
            srs_details=srs_details, 
            raw_eval_data=raw_eval_data
        )

        if success:
            print(f"  Successfully updated knowledge profile for concept '{concept_id}'.")
        else:
            print(f"  Failed to update knowledge profile for concept '{concept_id}'.")
        
        return success
