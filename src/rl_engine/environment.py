# src/rl_engine/environment.py
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
import json
import logging
from .session_manager import SessionManager
import asyncio

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

@runtime_checkable
class ProfileManagerProtocol(Protocol):
    """Protocol for profile manager interface"""
    
    def get_profile(self, learner_id: str):
        ...
    
    def create_profile(self, learner_id: str):
        ...
    
    def get_concept_knowledge(self, learner_id: str, concept_id: str):
        ...
    
    def get_last_attempted_concept_and_doc(self, learner_id: str):
        ...

@runtime_checkable
class AsyncProfileManagerProtocol(Protocol):
    """Protocol for async profile manager interface"""
    
    async def get_profile(self, learner_id: str):
        ...
    
    async def create_profile(self, learner_id: str):
        ...
    
    async def get_concept_knowledge(self, learner_id: str, concept_id: str):
        ...
    
    async def get_last_attempted_concept_and_doc(self, learner_id: str):
        ...

class ProfileManagerAdapter:
    """Adapter to handle both sync and async profile managers consistently"""
    
    def __init__(self, profile_manager):
        self.profile_manager = profile_manager
        self._is_async = self._detect_async_interface()
        logger.info(f"ProfileManagerAdapter initialized with {'async' if self._is_async else 'sync'} interface")
    
    def _detect_async_interface(self) -> bool:
        """Detect if the profile manager uses async interface"""
        # Check if the profile manager has async methods by looking for coroutine functions
        methods_to_check = ['get_profile', 'get_concept_knowledge', 'get_last_attempted_concept_and_doc']
        
        for method_name in methods_to_check:
            if hasattr(self.profile_manager, method_name):
                method = getattr(self.profile_manager, method_name)
                if asyncio.iscoroutinefunction(method):
                    return True
        
        return False
    
    async def get_profile(self, learner_id: str):
        """Get learner profile with proper async/sync handling"""
        if self._is_async:
            return await self.profile_manager.get_profile(learner_id)
        else:
            return self.profile_manager.get_profile(learner_id)
    
    async def create_profile(self, learner_id: str):
        """Create learner profile with proper async/sync handling"""
        if self._is_async:
            return await self.profile_manager.create_profile(learner_id)
        else:
            return self.profile_manager.create_profile(learner_id)
    
    async def get_concept_knowledge(self, learner_id: str, concept_id: str):
        """Get concept knowledge with proper async/sync handling"""
        if self._is_async:
            return await self.profile_manager.get_concept_knowledge(learner_id, concept_id)
        else:
            return self.profile_manager.get_concept_knowledge(learner_id, concept_id)
    
    async def get_last_attempted_concept_and_doc(self, learner_id: str):
        """Get last attempted concept with proper async/sync handling"""
        if self._is_async:
            return await self.profile_manager.get_last_attempted_concept_and_doc(learner_id)
        else:
            return self.profile_manager.get_last_attempted_concept_and_doc(learner_id)

class RLEnvironment:
    """
    RL Environment for adaptive question selection
    Manages state transitions and action space
    """
    
    def __init__(self, profile_manager, retriever):
        self.profile_manager = ProfileManagerAdapter(profile_manager)
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
            # FIXED: Use ProfileManagerAdapter for consistent async/sync handling
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
        
        # FIXED: Improved action filtering with better edge case handling and bounds checking
        for i, concept_id in enumerate(self.available_concepts):
            # FIXED: Validate concept index is within bounds
            if i >= len(self.available_concepts):
                logger.error(f"Concept index {i} out of bounds for {len(self.available_concepts)} concepts")
                continue
            
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
                    
                    # FIXED: Validate action index before adding
                    if 0 <= action_idx < self.action_space_size:
                        valid_actions.append(action_idx)
                    else:
                        logger.error(f"Generated invalid action index {action_idx} for concept {concept_id}, difficulty {difficulty_level}")
            
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
                    
                    # FIXED: Validate action index before adding
                    if 0 <= action_idx < self.action_space_size:
                        valid_actions.append(action_idx)
                    else:
                        logger.error(f"Generated invalid action index {action_idx} for concept {concept_id}, difficulty {difficulty_level}")
        
        # FIXED: Enhanced fallback mechanisms for edge cases
        if not valid_actions:
            logger.warning(f"No valid actions found for learner {state.learner_id}, checking fallback options")
            
            # Fallback 1: Allow all concepts with easy difficulty only
            for i, concept_id in enumerate(self.available_concepts):
                if i >= len(self.available_concepts):
                    continue
                    
                action_idx = i * len(DifficultyLevel) + list(DifficultyLevel).index(DifficultyLevel.EASY)
                
                # FIXED: Validate action index before adding
                if 0 <= action_idx < self.action_space_size:
                    valid_actions.append(action_idx)
                else:
                    logger.error(f"Fallback generated invalid action index {action_idx} for concept {concept_id}")
            
            if not valid_actions:
                logger.error(f"Still no valid actions after easy-only fallback for learner {state.learner_id}")
                # Fallback 2: Allow all actions (last resort)
                valid_actions = [i for i in range(self.action_space_size)]
                logger.warning(f"Using all actions as final fallback for learner {state.learner_id}")
        
        # FIXED: Final validation that we have valid actions
        if not valid_actions:
            logger.error(f"CRITICAL: No valid actions available for learner {state.learner_id}")
            raise ValueError(f"No valid actions available for learner {state.learner_id}")
        
        # FIXED: Final bounds check (should be redundant but safe)
        valid_actions = [action for action in valid_actions if 0 <= action < self.action_space_size]
        
        if not valid_actions:
            logger.error(f"All valid actions were out of bounds for learner {state.learner_id}")
            raise ValueError(f"All valid actions out of bounds for learner {state.learner_id}")
        
        logger.debug(f"Generated {len(valid_actions)} valid actions for learner {state.learner_id}")
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