# src/api/feedback_models.py
from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum

class FeedbackRating(str, Enum):
    UP = "up"
    DOWN = "down"

class InteractionFeedbackRequest(BaseModel):
    rating: FeedbackRating = Field(..., description="User's rating of the question: 'up' for helpful, 'down' for not helpful")

class InteractionFeedbackResponse(BaseModel):
    interaction_id: str
    learner_id: str
    rating: FeedbackRating
    reward_calculated: bool
    reward_total: Optional[float] = None
    reward_components: Optional[dict] = None
    message: str