# src/api/models.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class LearnerInteractionStartRequest(BaseModel):
    learner_id: str = Field(..., description="Unique identifier for the learner.")

class QuestionResponse(BaseModel):
    question_id: str = Field(..., description="Unique identifier for the question/concept.")
    concept_name: str = Field(..., description="Name of the concept the question is about.")
    question_text: str = Field(..., description="The text of the question presented to the learner.")
    context_for_evaluation: str = Field(..., description="The context provided with the question, used for evaluation.")
    # We might add difficulty_level here later if needed by the frontend

class AnswerSubmissionRequest(BaseModel):
    learner_id: str = Field(..., description="Unique identifier for the learner.")
    question_id: str = Field(..., description="Identifier of the question/concept being answered (from QuestionResponse).")
    question_text: str = Field(..., description="The original question text.")
    context_for_evaluation: str = Field(..., description="The context that was provided with the question.")
    learner_answer: str = Field(..., description="The answer submitted by the learner.")

class EvaluationResult(BaseModel):
    accuracy_score: float = Field(..., ge=0.0, le=1.0, description="Accuracy score from 0.0 to 1.0.")
    feedback: str = Field(..., description="Textual feedback on the answer.")
    # Optional: If your evaluator can provide a model/correct answer
    # correct_answer_suggestion: Optional[str] = Field(None, description="A suggested correct answer or key points.")

class AnswerSubmissionResponse(BaseModel):
    learner_id: str
    question_id: str # This is the concept_id
    evaluation: EvaluationResult
    # We can add more details here, like updated knowledge scores, if needed by the client.
    # For now, keeping it focused on the direct result of answer submission.

class ErrorResponse(BaseModel):
    detail: str
