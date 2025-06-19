# src/rl_engine/environment.py
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging

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
    session_length: int  # questions answered in current session
    recent_engagement: List[float]  # last 5 engagement scores
    time_since_last_question: float  # minutes
    
    def to_feature_vector(self, available_concepts: List[str]) -> np.ndarray:
        """Convert state to feature vector for RL agent"""
        features = []
        
        # Global learner features
        features.extend([
            self.last_score,
            self.session_length / 10.0,  # normalize
            np.mean(self.recent_engagement) if self.recent_engagement else 0.0,
            min(self.time_since_last_question / 30.0, 1.0),  # cap at 30 minutes
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
    
    @classmethod
    def get_feature_size(cls, num_concepts: int) -> int:
        """Calculate the size of the feature vector"""
        return 4 + (num_concepts * 3) + 3  # global + per-concept + difficulty

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
            
            logger.info(f"Action space size: {self.action_space_size}")
            logger.info(f"State space size: {self.state_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize RL environment: {e}")
            raise
    
    async def get_current_state(self, learner_id: str) -> State:
        """Get the current state for a learner"""
        try:
            # Get learner profile
            profile = await self.profile_manager.get_profile(learner_id)
            if not profile:
                await self.profile_manager.create_profile(learner_id)
            
            # Get concept knowledge
            concept_scores = {}
            concept_attempts = {}
            
            for concept_id in self.available_concepts:
                knowledge = await self.profile_manager.get_concept_knowledge(learner_id, concept_id)
                if knowledge:
                    concept_scores[concept_id] = knowledge.get('current_score', 0.0) / 10.0  # normalize to 0-1
                    concept_attempts[concept_id] = knowledge.get('total_attempts', 0)
                else:
                    concept_scores[concept_id] = 0.0
                    concept_attempts[concept_id] = 0
            
            # Get last attempted concept and score
            last_concept_id, last_doc_id = await self.profile_manager.get_last_attempted_concept_and_doc(learner_id)
            last_score = 0.0
            last_difficulty = None
            
            if last_concept_id:
                last_knowledge = await self.profile_manager.get_concept_knowledge(learner_id, last_concept_id)
                if last_knowledge:
                    last_score = last_knowledge.get('current_score', 0.0) / 10.0
                    difficulty_str = last_knowledge.get('current_difficulty_level', 'easy')
                    last_difficulty = DifficultyLevel(difficulty_str)
            
            # Calculate session length and recent engagement
            # This would need to be tracked separately in a session management system
            session_length = 0  # TODO: implement session tracking
            recent_engagement = []  # TODO: implement engagement tracking
            time_since_last_question = 0.0  # TODO: implement timing
            
            return State(
                learner_id=learner_id,
                concept_scores=concept_scores,
                concept_attempts=concept_attempts,
                last_concept_id=last_concept_id,
                last_score=last_score,
                last_difficulty=last_difficulty,
                session_length=session_length,
                recent_engagement=recent_engagement,
                time_since_last_question=time_since_last_question
            )
            
        except Exception as e:
            logger.error(f"Error getting current state for learner {learner_id}: {e}")
            raise
    
    def action_index_to_action(self, action_index: int) -> Action:
        """Convert action index to Action object"""
        concept_idx = action_index // len(DifficultyLevel)
        difficulty_idx = action_index % len(DifficultyLevel)
        
        concept_id = self.available_concepts[concept_idx]
        difficulty = list(DifficultyLevel)[difficulty_idx]
        
        return Action(concept_id=concept_id, difficulty_level=difficulty)
    
    def action_to_action_index(self, action: Action) -> int:
        """Convert Action object to action index"""
        concept_idx = self.available_concepts.index(action.concept_id)
        difficulty_idx = list(DifficultyLevel).index(action.difficulty_level)
        
        return concept_idx * len(DifficultyLevel) + difficulty_idx
    
    def get_valid_actions(self, state: State) -> List[int]:
        """Get list of valid action indices for current state"""
        valid_actions = []
        
        for i, concept_id in enumerate(self.available_concepts):
            # Skip concepts that are already mastered (score > 0.9)
            if state.concept_scores.get(concept_id, 0.0) > 0.9:
                continue
                
            # Add all difficulty levels for valid concepts
            for j in range(len(DifficultyLevel)):
                action_idx = i * len(DifficultyLevel) + j
                valid_actions.append(action_idx)
        
        return valid_actions if valid_actions else list(range(self.action_space_size))