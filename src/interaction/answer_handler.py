# src.interaction.answer_handler.py
import asyncio
from typing import Dict, Any, Optional
import os 
from src.evaluation.answer_evaluator import AnswerEvaluator
from src.learner_model.knowledge_tracker import KnowledgeTracker

class AnswerHandler:
    """
    Manages the process of a learner submitting an answer to a question,
    coordinating evaluation and profile updates.
    """

    def __init__(self, 
                 evaluator: AnswerEvaluator, 
                 tracker: KnowledgeTracker, 
                 question_generator=None):
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
                            question_id: str, 
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

        evaluation_result = await self.evaluator.evaluate_answer(
            question_text,
            retrieved_context,
            learner_answer
        )
        
        print(f"  Evaluation Result from Evaluator: {evaluation_result}")

        accuracy_score = evaluation_result.get("accuracy_score", 0.0) 
        feedback = evaluation_result.get("feedback", "No specific feedback provided.")
        correct_answer_suggestion = evaluation_result.get("correct_answer") # Get suggested correct answer

        await self.tracker.update_knowledge_level(
            learner_id=learner_id,
            concept_id=question_id, 
            accuracy_score=accuracy_score,
            raw_eval_data=evaluation_result # Store the full evaluation, including correct_answer if present
        )

        return {
            "learner_id": learner_id,
            "question_id": question_id, 
            "accuracy_score": accuracy_score, 
            "feedback": feedback,
            "correct_answer_suggestion": correct_answer_suggestion # Pass it along
        }

async def demo_answer_submission():
    print("--- AnswerHandler Demo (with real component setup) ---")
    import os
    from src import config
    from src.learner_model.profile_manager import LearnerProfileManager

    demo_db_path = os.path.join(config.DATA_DIR, "demo_handler_profiles_v2.sqlite3")
    if os.path.exists(demo_db_path): os.remove(demo_db_path)
    
    profile_manager = LearnerProfileManager(db_path=demo_db_path)
    real_evaluator = AnswerEvaluator() 
    real_tracker = KnowledgeTracker(profile_manager=profile_manager)
    answer_handler = AnswerHandler(evaluator=real_evaluator, tracker=real_tracker)

    learner_id = "learner_handler_001"
    profile_manager.create_profile(learner_id)

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
    
    profile_manager.close_db()
    print("\n--- AnswerHandler Demo Finished ---")

if __name__ == "__main__":
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        print(f"AnswerHandler Demo: Found .env file at {dotenv_path}, loading.")
        from dotenv import load_dotenv
        load_dotenv(dotenv_path)
    asyncio.run(demo_answer_submission())
