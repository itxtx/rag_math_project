# src/api/models.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class TopicResponse(BaseModel):
    doc_id: str = Field(..., description="Unique identifier for the topic/document.")
    title: str = Field(..., description="Title of the topic/document.")
    description: Optional[str] = Field(None, description="Optional description of the topic.")
    concepts: List[str] = Field(default_factory=list, description="List of concepts covered in this topic.")

class LearnerInteractionStartRequest(BaseModel):
    learner_id: str = Field(..., description="Unique identifier for the learner.")
    topic_id: Optional[str] = Field(None, description="Optional topic_id (doc_id) to focus question selection on a specific document.")

class QuestionResponse(BaseModel):
    question_id: str = Field(..., description="Unique identifier for the question/concept (this is the parent_block_id).")
    doc_id: str = Field(..., description="Identifier of the document/topic this question belongs to.") 
    concept_name: str = Field(..., description="Name of the concept the question is about.")
    question_text: str = Field(..., description="The text of the question presented to the learner.")
    context_for_evaluation: str = Field(..., description="The context provided with the question, used for evaluation. This is also the context to display if is_new_concept_context_presented is true.")
    is_new_concept_context_presented: Optional[bool] = Field(False, description="Flag indicating if the context (in context_for_evaluation) was just presented as new material to the learner.")

class AnswerSubmissionRequest(BaseModel):
    learner_id: str = Field(..., description="Unique identifier for the learner.")
    question_id: str = Field(..., description="Identifier of the question/concept being answered (from QuestionResponse).")
    doc_id: str = Field(..., description="Identifier of the document/topic this question belongs to.") 
    question_text: str = Field(..., description="The original question text.")
    context_for_evaluation: str = Field(..., description="The context that was provided with the question.")
    learner_answer: str = Field(..., description="The answer submitted by the learner.")

class EvaluationResult(BaseModel):
    accuracy_score: float = Field(..., ge=0.0, le=1.0, description="Accuracy score from 0.0 to 1.0.")
    feedback: str = Field(..., description="Textual feedback on the answer.")
    correct_answer: Optional[str] = None 

class AnswerSubmissionResponse(BaseModel):
    learner_id: str
    question_id: str 
    doc_id: str 
    evaluation: EvaluationResult

class ErrorResponse(BaseModel):
    detail: str
