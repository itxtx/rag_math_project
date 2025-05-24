# src.interaction.answer_handler.py
import asyncio
from typing import Dict, Any, Optional

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
                            question_id: str, # Concept ID for tracking
                            doc_id: str, # <<< ADDED doc_id parameter
                            question_text: str,
                            retrieved_context: str, 
                            learner_answer: str
                            ) -> Dict[str, Any]:
        """
        Processes a learner's submitted answer.
        """
        print(f"\nAnswerHandler: Learner '{learner_id}' submitted answer for question/concept '{question_id}' from doc '{doc_id}'.")
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
        correct_answer_suggestion = evaluation_result.get("correct_answer")

        await self.tracker.update_knowledge_level(
            learner_id=learner_id,
            concept_id=question_id, 
            doc_id=doc_id, # <<< PASS doc_id here
            accuracy_score=accuracy_score,
            raw_eval_data=evaluation_result 
        )

        return {
            "learner_id": learner_id,
            "question_id": question_id, 
            "doc_id": doc_id, # Optionally return doc_id if useful for client
            "accuracy_score": accuracy_score, 
            "feedback": feedback,
            "correct_answer_suggestion": correct_answer_suggestion
        }
