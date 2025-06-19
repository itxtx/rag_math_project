# src/rl_engine/environment.py
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging
from .session_manager import SessionManager

logger = logging.getLogger(__name__)

class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

@dataclass
class Action:
    """Represents an action the RL agent can take"""
    concept_id: str
    difficulty_level: DifficultyLevel
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "difficulty_level": self.difficulty_level.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Action':
        return cls(
            concept_id=data["concept_id"],
            difficulty_level=DifficultyLevel(data["difficulty_level"])
        )

@dataclass
class State:
    """Represents the learner's current knowledge state"""
    learner_id: str
    concept_scores: Dict[str, float]  # concept_id -> score (0-1)
    concept_attempts: Dict[str, int]  # concept_id -> number of attempts
    last_concept_id: Optional[str]
    last_score: float
    last_difficulty: Optional[DifficultyLevel]
    session_data: Dict[str, float]  # FIXED: Now uses real session data
    
    def to_feature_vector(self, available_concepts: List[str]) -> np.ndarray:
        """Convert state to feature vector for RL agent"""
        features = []
        
        # Global learner features - FIXED: Now uses real session data
        features.extend([
            self.last_score,
            self.session_data.get('session_length_normalized', 0.0),
            self.session_data.get('average_engagement', 0.5),
            self.session_data.get('time_since_last_normalized', 0.0),
        ])
        
        # Session-based features - FIXED: Added more sophisticated features
        features.extend([
            self.session_data.get('fatigue_factor', 1.0),
            self.session_data.get('momentum', 0.5),
            self.session_data.get('current_streak_normalized', 0.0),
            self.session_data.get('upvote_rate', 0.5),
        ])
        
        # Per-concept features (fixed size based on available concepts)
        for concept_id in sorted(available_concepts):
            score = self.concept_scores.get(concept_id, 0.0)
            attempts = min(self.concept_attempts.get(concept_id, 0) / 10.0, 1.0)  # normalize
            is_last = 1.0 if concept_id == self.last_concept_id else 0.0
            
            features.extend([score, attempts, is_last])
        
        # Last difficulty as one-hot
        difficulty_onehot = [0.0, 0.0, 0.0]
        if self.last_difficulty:
            if self.last_difficulty == DifficultyLevel.EASY:
                difficulty_onehot[0] = 1.0
            elif self.last_difficulty == DifficultyLevel.MEDIUM:
                difficulty_onehot[1] = 1.0
            elif self.last_difficulty == DifficultyLevel.HARD:
                difficulty_onehot[2] = 1.0
        
        features.extend(difficulty_onehot)
        
        return np.array(features, dtype=np.float32)
    
    def validate_feature_vector(self, available_concepts: List[str]) -> bool:
        """Validate that the feature vector has the correct size"""
        feature_vector = self.to_feature_vector(available_concepts)
        expected_size = self.get_feature_size(len(available_concepts))
        
        if len(feature_vector) != expected_size:
            logger.error(f"Feature vector size mismatch! Expected {expected_size}, got {len(feature_vector)}")
            logger.error(f"Available concepts: {len(available_concepts)}")
            return False
        
        return True
    
    @classmethod
    def get_feature_size(cls, num_concepts: int) -> int:
        """Calculate the size of the feature vector"""
        global_features = 4  # last_score, session_length, avg_engagement, time_since_last
        session_features = 4  # fatigue, momentum, streak, upvote_rate
        per_concept_features = num_concepts * 3  # score, attempts, is_last per concept
        difficulty_features = 3  # one-hot encoding for difficulty
        
        total = global_features + session_features + per_concept_features + difficulty_features
        return total

    @classmethod
    def get_feature_breakdown(cls, num_concepts: int) -> Dict[str, int]:
        """Get detailed breakdown of feature vector components"""
        return {
            'global_features': 4,  # last_score, session_length, avg_engagement, time_since_last
            'session_features': 4,  # fatigue, momentum, streak, upvote_rate
            'per_concept_features': num_concepts * 3,  # score, attempts, is_last per concept
            'difficulty_features': 3,  # one-hot encoding for difficulty
            'total': cls.get_feature_size(num_concepts)
        }

class RLEnvironment:
    """
    RL Environment for adaptive question selection
    Manages state transitions and action space
    """
    
    def __init__(self, profile_manager, retriever):
        self.profile_manager = profile_manager
        self.retriever = retriever
        self.available_concepts = []
        self.action_space_size = 0
        self.state_size = 0
        
        # FIXED: Added session manager integration
        self.session_manager = SessionManager(session_timeout_minutes=30)
        
    async def initialize(self):
        """Initialize the environment with available concepts"""
        logger.info("Initializing RL Environment...")
        
        # Get all available concepts from the curriculum
        try:
            all_documents = await self.retriever.get_all_documents()
            concepts = set()
            for doc in all_documents:
                if doc.get('parent_block_id'):
                    concepts.add(doc['parent_block_id'])
            
            self.available_concepts = sorted(list(concepts))
            logger.info(f"Found {len(self.available_concepts)} available concepts")
            
            # Calculate sizes
            self.action_space_size = len(self.available_concepts) * len(DifficultyLevel)
            self.state_size = State.get_feature_size(len(self.available_concepts))
            
            # Log detailed feature breakdown
            feature_breakdown = State.get_feature_breakdown(len(self.available_concepts))
            logger.info(f"Feature vector breakdown: {feature_breakdown}")
            logger.info(f"Action space size: {self.action_space_size}")
            logger.info(f"State space size: {self.state_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize RL environment: {e}")
            raise
    
    async def get_current_state(self, learner_id: str) -> State:
        """Get the current state for a learner with real session data"""
        try:
            # Get learner profile
            profile = self.profile_manager.get_profile(learner_id)  # FIXED: Removed await
            if not profile:
                self.profile_manager.create_profile(learner_id)  # FIXED: Removed await
            
            # Get concept knowledge
            concept_scores = {}
            concept_attempts = {}
            
            for concept_id in self.available_concepts:
                knowledge = self.profile_manager.get_concept_knowledge(learner_id, concept_id)  # FIXED: Removed await
                if knowledge:
                    concept_scores[concept_id] = knowledge.get('current_score', 0.0) / 10.0  # normalize to 0-1
                    concept_attempts[concept_id] = knowledge.get('total_attempts', 0)
                else:
                    concept_scores[concept_id] = 0.0
                    concept_attempts[concept_id] = 0
            
            # Get last attempted concept and score
            last_concept_id, last_doc_id = self.profile_manager.get_last_attempted_concept_and_doc(learner_id)  # FIXED: Removed await
            last_score = 0.0
            last_difficulty = None
            
            if last_concept_id:
                last_knowledge = self.profile_manager.get_concept_knowledge(learner_id, last_concept_id)  # FIXED: Removed await
                if last_knowledge:
                    last_score = last_knowledge.get('current_score', 0.0) / 10.0
                    difficulty_str = last_knowledge.get('current_difficulty_level', 'easy')
                    try:
                        last_difficulty = DifficultyLevel(difficulty_str)
                    except ValueError:
                        last_difficulty = DifficultyLevel.EASY
            
            # FIXED: Get real session data from session manager
            session_data = self.session_manager.get_session_state_for_rl(learner_id)
            
            state = State(
                learner_id=learner_id,
                concept_scores=concept_scores,
                concept_attempts=concept_attempts,
                last_concept_id=last_concept_id,
                last_score=last_score,
                last_difficulty=last_difficulty,
                session_data=session_data  # FIXED: Now uses real session data
            )
            
            # Validate feature vector size
            if not state.validate_feature_vector(self.available_concepts):
                logger.error(f"Feature vector validation failed for learner {learner_id}")
                raise ValueError(f"Feature vector size mismatch for learner {learner_id}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error getting current state for learner {learner_id}: {e}")
            raise
    
    def action_index_to_action(self, action_index: int) -> Action:
        """Convert action index to Action object"""
        if action_index < 0 or action_index >= self.action_space_size:
            raise ValueError(f"Invalid action index {action_index}, must be in range [0, {self.action_space_size})")
        
        concept_idx = action_index // len(DifficultyLevel)
        difficulty_idx = action_index % len(DifficultyLevel)
        
        if concept_idx >= len(self.available_concepts):
            raise ValueError(f"Invalid concept index {concept_idx}, only {len(self.available_concepts)} concepts available")
        
        concept_id = self.available_concepts[concept_idx]
        difficulty = list(DifficultyLevel)[difficulty_idx]
        
        return Action(concept_id=concept_id, difficulty_level=difficulty)
    
    def action_to_action_index(self, action: Action) -> int:
        """Convert Action object to action index"""
        try:
            concept_idx = self.available_concepts.index(action.concept_id)
            difficulty_idx = list(DifficultyLevel).index(action.difficulty_level)
            
            return concept_idx * len(DifficultyLevel) + difficulty_idx
        except ValueError as e:
            logger.error(f"Invalid action: {action}, error: {e}")
            raise
    
    def get_valid_actions(self, state: State) -> List[int]:
        """Get list of valid action indices for current state"""
        valid_actions = []
        
        # FIXED: Improved action filtering based on learner state
        for i, concept_id in enumerate(self.available_concepts):
            concept_score = state.concept_scores.get(concept_id, 0.0)
            concept_attempts = state.concept_attempts.get(concept_id, 0)
            
            # Skip concepts that are fully mastered (score > 0.95)
            if concept_score > 0.95:
                continue
            
            # For new concepts (no attempts), only allow easy and medium difficulty
            if concept_attempts == 0:
                for difficulty_level in [DifficultyLevel.EASY, DifficultyLevel.MEDIUM]:
                    difficulty_idx = list(DifficultyLevel).index(difficulty_level)
                    action_idx = i * len(DifficultyLevel) + difficulty_idx
                    valid_actions.append(action_idx)
            
            # For concepts with some attempts, allow all difficulties based on performance
            else:
                if concept_score < 0.3:
                    # Low performance - prefer easier questions
                    allowed_difficulties = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM]
                elif concept_score < 0.7:
                    # Medium performance - allow medium and hard
                    allowed_difficulties = [DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
                else:
                    # High performance - allow all difficulties
                    allowed_difficulties = list(DifficultyLevel)
                
                for difficulty_level in allowed_difficulties:
                    difficulty_idx = list(DifficultyLevel).index(difficulty_level)
                    action_idx = i * len(DifficultyLevel) + difficulty_idx
                    valid_actions.append(action_idx)
        
        # Fallback - if no valid actions, allow all actions
        if not valid_actions:
            logger.warning(f"No valid actions found for learner {state.learner_id}, allowing all actions")
            valid_actions = list(range(self.action_space_size))
        
        return valid_actions
    
    def start_interaction(self, learner_id: str, interaction_id: str):
        """Notify session manager of interaction start"""
        self.session_manager.start_interaction(learner_id, interaction_id)
    
    def complete_interaction(self, learner_id: str, interaction_id: str, was_correct: bool, vote_type: str = None):
        """Notify session manager of interaction completion"""
        self.session_manager.complete_interaction(learner_id, interaction_id, was_correct, vote_type)
    
    def cleanup_sessions(self):
        """Clean up expired sessions"""
        self.session_manager.cleanup_expired_sessions()
    
    def get_session_stats(self, learner_id: str) -> Dict[str, Any]:
        """Get session statistics for a learner"""
        return self.session_manager.get_session_stats(learner_id)
    
    def get_environment_stats(self) -> Dict[str, Any]:
        """Get comprehensive environment statistics"""
        session_stats = self.session_manager.get_global_stats()
        
        return {
            'available_concepts': len(self.available_concepts),
            'action_space_size': self.action_space_size,
            'state_size': self.state_size,
            'session_stats': session_stats,
            'environment_initialized': len(self.available_concepts) > 0
        }