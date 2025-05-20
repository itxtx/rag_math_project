# src.interaction.answer_handler.py
import asyncio
from typing import Dict, Any, Optional
import os 
# Import the actual modules
from src.evaluation.answer_evaluator import AnswerEvaluator
from src.learner_model.knowledge_tracker import KnowledgeTracker
# from src.generation import question_generator_rag # If it needs to fetch questions

class AnswerHandler:
    """
    Manages the process of a learner submitting an answer to a question,
    coordinating evaluation and profile updates.
    """

    def __init__(self, 
                 evaluator: AnswerEvaluator, 
                 tracker: KnowledgeTracker, 
                 question_generator=None): # question_generator remains optional
        """
        Initializes the AnswerHandler.

        Args:
            evaluator: An instance of the AnswerEvaluator.
            tracker: An instance of the KnowledgeTracker.
            question_generator: An optional instance of a question generator.
        """
        if not isinstance(evaluator, AnswerEvaluator):
            raise TypeError("evaluator must be an instance of AnswerEvaluator.")
        if not isinstance(tracker, KnowledgeTracker):
            raise TypeError("tracker must be an instance of KnowledgeTracker.")
            
        self.evaluator = evaluator
        self.tracker = tracker
        self.question_generator = question_generator 
        
        print("AnswerHandler initialized with real components.")

    async def submit_answer(self,
                            learner_id: str,
                            question_id: str, # Concept ID for tracking
                            question_text: str,
                            retrieved_context: str, 
                            learner_answer: str
                            ) -> Dict[str, Any]:
        """
        Processes a learner's submitted answer.
        """
        print(f"\nAnswerHandler: Learner '{learner_id}' submitted answer for question/concept '{question_id}'.")
        print(f"  Question: \"{question_text}\"")
        print(f"  Learner's Answer: \"{learner_answer}\"")

        # Step 1: Evaluate the answer using the real evaluator
        evaluation_result = await self.evaluator.evaluate_answer(
            question_text,
            retrieved_context,
            learner_answer
        )
        
        print(f"  Evaluation Result: {evaluation_result}")

        accuracy_score = evaluation_result.get("accuracy_score", 0.0) 
        feedback = evaluation_result.get("feedback", "No specific feedback provided.")
        
        # Step 2: Update learner's knowledge profile using the real tracker
        # The tracker will calculate the graded score and determine 'answered_correctly'
        await self.tracker.update_knowledge_level(
            learner_id=learner_id,
            concept_id=question_id, 
            accuracy_score=accuracy_score, # Pass the 0-1 accuracy score
            raw_eval_data=evaluation_result 
        )

        # For the return value, let's also include the graded score calculated by the tracker
        # For simplicity, we can recalculate it here or assume tracker might return it.
        # Let's assume the tracker's internal logic is what matters for the DB.
        # The handler can return the direct evaluation.
        return {
            "learner_id": learner_id,
            "question_id": question_id, # This is the concept_id
            "accuracy_score": accuracy_score, # 0.0 - 1.0
            "feedback": feedback,
            # "graded_score" and "answered_correctly" are handled by the tracker internally for DB update
        }

# No mock classes needed here anymore if this module is always instantiated with real ones.

async def demo_answer_submission():
    print("--- AnswerHandler Demo (with real component setup) ---")
    
    # For this demo to run, you need instances of actual components.
    # This requires setting up ProfileManager first for the KnowledgeTracker.
    import os
    from src import config
    from src.learner_model.profile_manager import LearnerProfileManager

    # Setup ProfileManager for the demo
    demo_db_path = os.path.join(config.DATA_DIR, "demo_handler_profiles.sqlite3")
    if os.path.exists(demo_db_path):
        os.remove(demo_db_path)
    
    profile_manager = LearnerProfileManager(db_path=demo_db_path)
    
    # Instantiate actual components
    # Ensure GEMINI_API_KEY is available in config or .env for AnswerEvaluator
    real_evaluator = AnswerEvaluator() 
    real_tracker = KnowledgeTracker(profile_manager=profile_manager)
    
    answer_handler = AnswerHandler(evaluator=real_evaluator, tracker=real_tracker)

    learner_id = "learner_handler_001"
    profile_manager.create_profile(learner_id) # Ensure profile exists

    question_id = "thermo_q_demo" 
    question_text = "What is the first law of thermodynamics?"
    retrieved_context = (
        "The first law of thermodynamics, also known as the law of conservation of energy, "
        "states that energy cannot be created or destroyed in an isolated system. "
        "It can only be transformed from one form to another."
    )
    
    learner_answer_good = "The first law of thermodynamics is about the conservation of energy."

    print("\nSubmitting an answer:")
    result = await answer_handler.submit_answer(
        learner_id, question_id, question_text, retrieved_context, learner_answer_good
    )
    print(f"Handler response: {result}")

    # Check profile manager for updates
    knowledge = profile_manager.get_concept_knowledge(learner_id, question_id)
    print(f"\nKnowledge for '{question_id}' after submission: {knowledge}")
    history = profile_manager.get_score_history(learner_id, question_id)
    print(f"Score history for '{question_id}': {history}")
    
    profile_manager.close_db()
    print("\n--- AnswerHandler Demo Finished ---")


if __name__ == "__main__":
    # Load .env for direct script execution if needed
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        print(f"AnswerHandler Demo: Found .env file at {dotenv_path}, loading.")
        from dotenv import load_dotenv
        load_dotenv(dotenv_path)

    asyncio.run(demo_answer_submission())
