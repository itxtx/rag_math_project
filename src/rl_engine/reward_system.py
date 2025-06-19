# src/rl_engine/reward_system.py
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class VoteType(Enum):
    UPVOTE = "up"
    DOWNVOTE = "down"
    TIMEOUT = "timeout"

@dataclass
class RewardComponents:
    """Components of the hybrid reward function"""
    vote_reward: float
    learning_reward: float
    effort_reward: float
    total_reward: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'vote_reward': self.vote_reward,
            'learning_reward': self.learning_reward,
            'effort_reward': self.effort_reward,
            'total_reward': self.total_reward
        }

class HybridRewardCalculator:
    """
    Calculates hybrid rewards based on engagement, learning progress, and effort
    Formula: R = (w_vote * R_vote) + (w_learn * R_learn) + (w_effort * R_effort)
    """
    
    def __init__(self, 
                 w_vote: float = 0.6,
                 w_learn: float = 0.3,
                 w_effort: float = 0.1):
        """
        Initialize reward calculator with component weights
        
        Args:
            w_vote: Weight for user engagement/vote component
            w_learn: Weight for learning progress component  
            w_effort: Weight for effort/difficulty component
        """
        self.w_vote = w_vote
        self.w_learn = w_learn
        self.w_effort = w_effort
        
        # Validate weights sum to 1.0
        total_weight = w_vote + w_learn + w_effort
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Reward weights sum to {total_weight}, not 1.0. Normalizing...")
            self.w_vote = w_vote / total_weight
            self.w_learn = w_learn / total_weight
            self.w_effort = w_effort / total_weight
        
        logger.info(f"Reward calculator initialized with weights: "
                   f"vote={self.w_vote:.2f}, learn={self.w_learn:.2f}, effort={self.w_effort:.2f}")
    
    def calculate_vote_reward(self, vote_type: VoteType) -> float:
        """
        Calculate reward based on user engagement feedback
        
        Args:
            vote_type: Type of user feedback
            
        Returns:
            R_vote: Vote reward component
        """
        if vote_type == VoteType.UPVOTE:
            return 1.0
        elif vote_type == VoteType.DOWNVOTE:
            return -1.0
        elif vote_type == VoteType.TIMEOUT:
            return -0.1
        else:
            logger.warning(f"Unknown vote type: {vote_type}")
            return 0.0
    
    def calculate_learning_reward(self, 
                                score_before: float, 
                                score_after: float) -> float:
        """
        Calculate reward based on learning progress
        
        Args:
            score_before: Knowledge score before question (0-1 scale)
            score_after: Knowledge score after question (0-1 scale)
            
        Returns:
            R_learn: Learning reward component
        """
        # Ensure scores are in valid range
        score_before = max(0.0, min(1.0, score_before))
        score_after = max(0.0, min(1.0, score_after))
        
        mastery_gain = score_after - score_before
        
        # Handle edge case where score_before is 1.0 (perfect mastery)
        if score_before >= 0.999:
            # If already mastered, small penalty for continuing to ask
            return -0.1 if mastery_gain <= 0 else 0.0
        
        # Normalize by remaining learning potential
        denominator = 1.0 - score_before
        r_learn = mastery_gain / denominator
        
        # Clip to reasonable range
        r_learn = max(-1.0, min(1.0, r_learn))
        
        return r_learn
    
    def calculate_effort_reward(self, difficulty: str) -> float:
        """
        Calculate reward based on question difficulty
        
        Args:
            difficulty: Question difficulty level ("easy", "medium", "hard")
            
        Returns:
            R_effort: Effort reward component
        """
        difficulty_mapping = {
            "easy": 0.1,
            "medium": 0.5,
            "hard": 1.0
        }
        
        return difficulty_mapping.get(difficulty.lower(), 0.1)
    
    def calculate_hybrid_reward(self,
                              vote_type: VoteType,
                              score_before: float,
                              score_after: float,
                              difficulty: str) -> RewardComponents:
        """
        Calculate the complete hybrid reward
        
        Args:
            vote_type: User engagement feedback
            score_before: Knowledge score before question (0-1)
            score_after: Knowledge score after question (0-1)
            difficulty: Question difficulty level
            
        Returns:
            RewardComponents: All reward components and total
        """
        # Calculate individual components
        r_vote = self.calculate_vote_reward(vote_type)
        r_learn = self.calculate_learning_reward(score_before, score_after)
        r_effort = self.calculate_effort_reward(difficulty)
        
        # Calculate weighted total
        total_reward = (self.w_vote * r_vote + 
                       self.w_learn * r_learn + 
                       self.w_effort * r_effort)
        
        components = RewardComponents(
            vote_reward=r_vote,
            learning_reward=r_learn,
            effort_reward=r_effort,
            total_reward=total_reward
        )
        
        logger.debug(f"Reward calculation: vote={r_vote:.3f}, learn={r_learn:.3f}, "
                    f"effort={r_effort:.3f}, total={total_reward:.3f}")
        
        return components

class InteractionTracker:
    """
    Tracks interactions and manages reward calculation timing
    """
    
    def __init__(self):
        self.pending_interactions = {}  # interaction_id -> interaction_data
        self.completed_interactions = {}  # interaction_id -> reward_data
        self.timeout_duration = timedelta(minutes=5)  # 5 minute timeout
    
    def start_interaction(self, 
                         interaction_id: str,
                         learner_id: str,
                         concept_id: str,
                         difficulty: str,
                         score_before: float,
                         state_before: np.ndarray) -> None:
        """
        Start tracking a new interaction
        
        Args:
            interaction_id: Unique identifier for the interaction
            learner_id: ID of the learner
            concept_id: ID of the concept being learned
            difficulty: Difficulty level of the question
            score_before: Knowledge score before the question
            state_before: RL state before the question
        """
        self.pending_interactions[interaction_id] = {
            'learner_id': learner_id,
            'concept_id': concept_id,
            'difficulty': difficulty,
            'score_before': score_before,
            'state_before': state_before,
            'start_time': datetime.now(),
            'completed': False
        }
        
        logger.info(f"Started tracking interaction {interaction_id} for learner {learner_id}")
    
    def complete_interaction(self,
                           interaction_id: str,
                           score_after: float,
                           state_after: np.ndarray,
                           vote_type: Optional[VoteType] = None) -> Optional[RewardComponents]:
        """
        Complete an interaction and calculate reward
        
        Args:
            interaction_id: ID of the interaction to complete
            score_after: Knowledge score after the question
            state_after: RL state after the question
            vote_type: User feedback (None = timeout)
            
        Returns:
            RewardComponents if successful, None if interaction not found
        """
        if interaction_id not in self.pending_interactions:
            logger.error(f"Interaction {interaction_id} not found in pending interactions")
            return None
        
        interaction_data = self.pending_interactions[interaction_id]
        
        # Handle timeout case
        if vote_type is None:
            elapsed_time = datetime.now() - interaction_data['start_time']
            if elapsed_time > self.timeout_duration:
                vote_type = VoteType.TIMEOUT
                logger.info(f"Interaction {interaction_id} timed out after {elapsed_time}")
            else:
                logger.warning(f"Interaction {interaction_id} completed without vote before timeout")
                vote_type = VoteType.TIMEOUT
        
        # Calculate reward
        calculator = HybridRewardCalculator()
        reward_components = calculator.calculate_hybrid_reward(
            vote_type=vote_type,
            score_before=interaction_data['score_before'],
            score_after=score_after,
            difficulty=interaction_data['difficulty']
        )
        
        # Store completed interaction data
        self.completed_interactions[interaction_id] = {
            **interaction_data,
            'score_after': score_after,
            'state_after': state_after,
            'vote_type': vote_type,
            'reward_components': reward_components,
            'completion_time': datetime.now(),
            'completed': True
        }
        
        # Remove from pending
        del self.pending_interactions[interaction_id]
        
        logger.info(f"Completed interaction {interaction_id} with reward {reward_components.total_reward:.3f}")
        
        return reward_components
    
    def get_pending_interactions(self) -> Dict[str, Dict]:
        """Get all pending interactions"""
        return self.pending_interactions.copy()
    
    def get_completed_interactions(self) -> Dict[str, Dict]:
        """Get all completed interactions"""
        return self.completed_interactions.copy()
    
    def get_interaction_data(self, interaction_id: str) -> Optional[Dict]:
        """Get data for a specific interaction"""
        if interaction_id in self.pending_interactions:
            return self.pending_interactions[interaction_id]
        elif interaction_id in self.completed_interactions:
            return self.completed_interactions[interaction_id]
        else:
            return None
    
    def cleanup_old_interactions(self, max_age_hours: int = 24):
        """Remove old completed interactions to prevent memory bloat"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clean up completed interactions
        to_remove = []
        for interaction_id, data in self.completed_interactions.items():
            if data.get('completion_time', datetime.now()) < cutoff_time:
                to_remove.append(interaction_id)
        
        for interaction_id in to_remove:
            del self.completed_interactions[interaction_id]
        
        # Handle very old pending interactions (force timeout)
        to_timeout = []
        for interaction_id, data in self.pending_interactions.items():
            if data['start_time'] < cutoff_time:
                to_timeout.append(interaction_id)
        
        for interaction_id in to_timeout:
            logger.warning(f"Force timing out old pending interaction {interaction_id}")
            # Get dummy state_after (same as state_before)
            state_after = self.pending_interactions[interaction_id]['state_before']
            score_after = self.pending_interactions[interaction_id]['score_before']
            self.complete_interaction(interaction_id, score_after, state_after, VoteType.TIMEOUT)
        
        if to_remove or to_timeout:
            logger.info(f"Cleaned up {len(to_remove)} completed and {len(to_timeout)} timed out interactions")

class RewardSystemManager:
    """
    High-level manager for the reward system
    Integrates with the RL training pipeline
    """
    
    def __init__(self, profile_manager):
        self.profile_manager = profile_manager
        self.interaction_tracker = InteractionTracker()
        self.reward_calculator = HybridRewardCalculator()
    
    async def start_interaction(self,
                              interaction_id: str,
                              learner_id: str,
                              concept_id: str,
                              difficulty: str,
                              state_before: np.ndarray) -> bool:
        """
        Start tracking a new interaction for reward calculation
        
        Returns:
            bool: Success status
        """
        try:
            # Get current knowledge score
            knowledge = await self.profile_manager.get_concept_knowledge(learner_id, concept_id)
            score_before = (knowledge.get('current_score', 0.0) / 10.0) if knowledge else 0.0
            
            self.interaction_tracker.start_interaction(
                interaction_id=interaction_id,
                learner_id=learner_id,
                concept_id=concept_id,
                difficulty=difficulty,
                score_before=score_before,
                state_before=state_before
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting interaction tracking: {e}")
            return False
    
    async def complete_interaction(self,
                                 interaction_id: str,
                                 learner_id: str,
                                 concept_id: str,
                                 state_after: np.ndarray,
                                 vote_type: Optional[VoteType] = None) -> Optional[Tuple[RewardComponents, np.ndarray, int, float, np.ndarray, bool]]:
        """
        Complete an interaction and get the RL training tuple
        
        Returns:
            Tuple of (reward_components, state_before, action, reward, state_after, done) or None
        """
        try:
            # Get updated knowledge score
            knowledge = await self.profile_manager.get_concept_knowledge(learner_id, concept_id)
            score_after = (knowledge.get('current_score', 0.0) / 10.0) if knowledge else 0.0
            
            # Complete the interaction
            reward_components = self.interaction_tracker.complete_interaction(
                interaction_id=interaction_id,
                score_after=score_after,
                state_after=state_after,
                vote_type=vote_type
            )
            
            if reward_components is None:
                return None
            
            # Get interaction data for RL tuple
            interaction_data = self.interaction_tracker.get_interaction_data(interaction_id)
            if not interaction_data:
                return None
            
            # Construct RL experience tuple
            state_before = interaction_data['state_before']
            reward = reward_components.total_reward
            done = False  # Episodes don't end with single questions
            
            # Action needs to be reconstructed from concept_id and difficulty
            # This would need to be stored in the interaction data or reconstructed
            action = 0  # Placeholder - needs proper implementation
            
            return (reward_components, state_before, action, reward, state_after, done)
            
        except Exception as e:
            logger.error(f"Error completing interaction: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reward system statistics"""
        pending = self.interaction_tracker.get_pending_interactions()
        completed = self.interaction_tracker.get_completed_interactions()
        
        # Calculate average rewards
        if completed:
            total_rewards = [data['reward_components'].total_reward for data in completed.values()]
            avg_reward = np.mean(total_rewards)
            reward_std = np.std(total_rewards)
        else:
            avg_reward = 0.0
            reward_std = 0.0
        
        return {
            'pending_interactions': len(pending),
            'completed_interactions': len(completed),
            'average_reward': avg_reward,
            'reward_std': reward_std,
            'reward_weights': {
                'vote': self.reward_calculator.w_vote,
                'learn': self.reward_calculator.w_learn,
                'effort': self.reward_calculator.w_effort
            }
        }
    
    def cleanup(self):
        """Cleanup old interactions"""
        self.interaction_tracker.cleanup_old_interactions()# src/rl_engine/reward_system.py
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class VoteType(Enum):
    UPVOTE = "up"
    DOWNVOTE = "down"
    TIMEOUT = "timeout"

@dataclass
class RewardComponents:
    """Components of the hybrid reward function"""
    vote_reward: float
    learning_reward: float
    effort_reward: float
    total_reward: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'vote_reward': self.vote_reward,
            'learning_reward': self.learning_reward,
            'effort_reward': self.effort_reward,
            'total_reward': self.total_reward
        }

class HybridRewardCalculator:
    """
    Calculates hybrid rewards based on engagement, learning progress, and effort
    Formula: R = (w_vote * R_vote) + (w_learn * R_learn) + (w_effort * R_effort)
    """
    
    def __init__(self, 
                 w_vote: float = 0.6,
                 w_learn: float = 0.3,
                 w_effort: float = 0.1):
        """
        Initialize reward calculator with component weights
        
        Args:
            w_vote: Weight for user engagement/vote component
            w_learn: Weight for learning progress component  
            w_effort: Weight for effort/difficulty component
        """
        self.w_vote = w_vote
        self.w_learn = w_learn
        self.w_effort = w_effort
        
        # Validate weights sum to 1.0
        total_weight = w_vote + w_learn + w_effort
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Reward weights sum to {total_weight}, not 1.0. Normalizing...")
            self.w_vote = w_vote / total_weight
            self.w_learn = w_learn / total_weight
            self.w_effort = w_effort / total_weight
        
        logger.info(f"Reward calculator initialized with weights: "
                   f"vote={self.w_vote:.2f}, learn={self.w_learn:.2f}, effort={self.w_effort:.2f}")
    
    def calculate_vote_reward(self, vote_type: VoteType) -> float:
        """
        Calculate reward based on user engagement feedback
        
        Args:
            vote_type: Type of user feedback
            
        Returns:
            R_vote: Vote reward component
        """
        if vote_type == VoteType.UPVOTE:
            return 1.0
        elif vote_type == VoteType.DOWNVOTE:
            return -1.0
        elif vote_type == VoteType.TIMEOUT:
            return -0.1
        else:
            logger.warning(f"Unknown vote type: {vote_type}")
            return 0.0
    
    def calculate_learning_reward(self, 
                                score_before: float, 
                                score_after: float) -> float:
        """
        Calculate reward based on learning progress
        
        Args:
            score_before: Knowledge score before question (0-1 scale)
            score_after: Knowledge score after question (0-1 scale)
            
        Returns:
            R_learn: Learning reward component
        """
        # Ensure scores are in valid range
        score_before = max(0.0, min(1.0, score_before))
        score_after = max(0.0, min(1.0, score_after))
        
        mastery_gain = score_after - score_before
        
        # Handle edge case where score_before is 1.0 (perfect mastery)
        if score_before >= 0.999:
            # If already mastered, small penalty for continuing to ask
            return -0.1 if mastery_gain <= 0 else 0.0
        
        # Normalize by remaining learning potential
        denominator = 1.0 - score_before
        r_learn = mastery_gain / denominator
        
        # Clip to reasonable range
        r_learn = max(-1.0, min(1.0, r_learn))
        
        return r_learn
    
    def calculate_effort_reward(self, difficulty: str) -> float:
        """
        Calculate reward based on question difficulty
        
        Args:
            difficulty: Question difficulty level ("easy", "medium", "hard")
            
        Returns:
            R_effort: Effort reward component
        """
        difficulty_mapping = {
            "easy": 0.1,
            "medium": 0.5,
            "hard": 1.0
        }
        
        return difficulty_mapping.get(difficulty.lower(), 0.1)
    
    def calculate_hybrid_reward(self,
                              vote_type: VoteType,
                              score_before: float,
                              score_after: float,
                              difficulty: str) -> RewardComponents:
        """
        Calculate the complete hybrid reward
        
        Args:
            vote_type: User engagement feedback
            score_before: Knowledge score before question (0-1)
            score_after: Knowledge score after question (0-1)
            difficulty: Question difficulty level
            
        Returns:
            RewardComponents: All reward components and total
        """
        # Calculate individual components
        r_vote = self.calculate_vote_reward(vote_type)
        r_learn = self.calculate_learning_reward(score_before, score_after)
        r_effort = self.calculate_effort_reward(difficulty)
        
        # Calculate weighted total
        total_reward = (self.w_vote * r_vote + 
                       self.w_learn * r_learn + 
                       self.w_effort * r_effort)
        
        components = RewardComponents(
            vote_reward=r_vote,
            learning_reward=r_learn,
            effort_reward=r_effort,
            total_reward=total_reward
        )
        
        logger.debug(f"Reward calculation: vote={r_vote:.3f}, learn={r_learn:.3f}, "
                    f"effort={r_effort:.3f}, total={total_reward:.3f}")
        
        return components

class InteractionTracker:
    """
    Tracks interactions and manages reward calculation timing
    """
    
    def __init__(self):
        self.pending_interactions = {}  # interaction_id -> interaction_data
        self.completed_interactions = {}  # interaction_id -> reward_data
        self.timeout_duration = timedelta(minutes=5)  # 5 minute timeout
    
    def start_interaction(self, 
                         interaction_id: str,
                         learner_id: str,
                         concept_id: str,
                         difficulty: str,
                         score_before: float,
                         state_before: np.ndarray) -> None:
        """
        Start tracking a new interaction
        
        Args:
            interaction_id: Unique identifier for the interaction
            learner_id: ID of the learner
            concept_id: ID of the concept being learned
            difficulty: Difficulty level of the question
            score_before: Knowledge score before the question
            state_before: RL state before the question
        """
        self.pending_interactions[interaction_id] = {
            'learner_id': learner_id,
            'concept_id': concept_id,
            'difficulty': difficulty,
            'score_before': score_before,
            'state_before': state_before,
            'start_time': datetime.now(),
            'completed': False
        }
        
        logger.info(f"Started tracking interaction {interaction_id} for learner {learner_id}")
    
    def complete_interaction(self,
                           interaction_id: str,
                           score_after: float,
                           state_after: np.ndarray,
                           vote_type: Optional[VoteType] = None) -> Optional[RewardComponents]:
        """
        Complete an interaction and calculate reward
        
        Args:
            interaction_id: ID of the interaction to complete
            score_after: Knowledge score after the question
            state_after: RL state after the question
            vote_type: User feedback (None = timeout)
            
        Returns:
            RewardComponents if successful, None if interaction not found
        """
        if interaction_id not in self.pending_interactions:
            logger.error(f"Interaction {interaction_id} not found in pending interactions")
            return None
        
        interaction_data = self.pending_interactions[interaction_id]
        
        # Handle timeout case
        if vote_type is None:
            elapsed_time = datetime.now() - interaction_data['start_time']
            if elapsed_time > self.timeout_duration:
                vote_type = VoteType.TIMEOUT
                logger.info(f"Interaction {interaction_id} timed out after {elapsed_time}")
            else:
                logger.warning(f"Interaction {interaction_id} completed without vote before timeout")
                vote_type = VoteType.TIMEOUT
        
        # Calculate reward
        calculator = HybridRewardCalculator()
        reward_components = calculator.calculate_hybrid_reward(
            vote_type=vote_type,
            score_before=interaction_data['score_before'],
            score_after=score_after,
            difficulty=interaction_data['difficulty']
        )
        
        # Store completed interaction data
        self.completed_interactions[interaction_id] = {
            **interaction_data,
            'score_after': score_after,
            'state_after': state_after,
            'vote_type': vote_type,
            'reward_components': reward_components,
            'completion_time': datetime.now(),
            'completed': True
        }
        
        # Remove from pending
        del self.pending_interactions[interaction_id]
        
        logger.info(f"Completed interaction {interaction_id} with reward {reward_components.total_reward:.3f}")
        
        return reward_components
    
    def get_pending_interactions(self) -> Dict[str, Dict]:
        """Get all pending interactions"""
        return self.pending_interactions.copy()
    
    def get_completed_interactions(self) -> Dict[str, Dict]:
        """Get all completed interactions"""
        return self.completed_interactions.copy()
    
    def get_interaction_data(self, interaction_id: str) -> Optional[Dict]:
        """Get data for a specific interaction"""
        if interaction_id in self.pending_interactions:
            return self.pending_interactions[interaction_id]
        elif interaction_id in self.completed_interactions:
            return self.completed_interactions[interaction_id]
        else:
            return None
    
    def cleanup_old_interactions(self, max_age_hours: int = 24):
        """Remove old completed interactions to prevent memory bloat"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clean up completed interactions
        to_remove = []
        for interaction_id, data in self.completed_interactions.items():
            if data.get('completion_time', datetime.now()) < cutoff_time:
                to_remove.append(interaction_id)
        
        for interaction_id in to_remove:
            del self.completed_interactions[interaction_id]
        
        # Handle very old pending interactions (force timeout)
        to_timeout = []
        for interaction_id, data in self.pending_interactions.items():
            if data['start_time'] < cutoff_time:
                to_timeout.append(interaction_id)
        
        for interaction_id in to_timeout:
            logger.warning(f"Force timing out old pending interaction {interaction_id}")
            # Get dummy state_after (same as state_before)
            state_after = self.pending_interactions[interaction_id]['state_before']
            score_after = self.pending_interactions[interaction_id]['score_before']
            self.complete_interaction(interaction_id, score_after, state_after, VoteType.TIMEOUT)
        
        if to_remove or to_timeout:
            logger.info(f"Cleaned up {len(to_remove)} completed and {len(to_timeout)} timed out interactions")

class RewardSystemManager:
    """
    High-level manager for the reward system
    Integrates with the RL training pipeline
    """
    
    def __init__(self, profile_manager):
        self.profile_manager = profile_manager
        self.interaction_tracker = InteractionTracker()
        self.reward_calculator = HybridRewardCalculator()
    
    async def start_interaction(self,
                              interaction_id: str,
                              learner_id: str,
                              concept_id: str,
                              difficulty: str,
                              state_before: np.ndarray) -> bool:
        """
        Start tracking a new interaction for reward calculation
        
        Returns:
            bool: Success status
        """
        try:
            # Get current knowledge score
            knowledge = await self.profile_manager.get_concept_knowledge(learner_id, concept_id)
            score_before = (knowledge.get('current_score', 0.0) / 10.0) if knowledge else 0.0
            
            self.interaction_tracker.start_interaction(
                interaction_id=interaction_id,
                learner_id=learner_id,
                concept_id=concept_id,
                difficulty=difficulty,
                score_before=score_before,
                state_before=state_before
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting interaction tracking: {e}")
            return False
    
    async def complete_interaction(self,
                                 interaction_id: str,
                                 learner_id: str,
                                 concept_id: str,
                                 state_after: np.ndarray,
                                 vote_type: Optional[VoteType] = None) -> Optional[Tuple[RewardComponents, np.ndarray, int, float, np.ndarray, bool]]:
        """
        Complete an interaction and get the RL training tuple
        
        Returns:
            Tuple of (reward_components, state_before, action, reward, state_after, done) or None
        """
        try:
            # Get updated knowledge score
            knowledge = await self.profile_manager.get_concept_knowledge(learner_id, concept_id)
            score_after = (knowledge.get('current_score', 0.0) / 10.0) if knowledge else 0.0
            
            # Complete the interaction
            reward_components = self.interaction_tracker.complete_interaction(
                interaction_id=interaction_id,
                score_after=score_after,
                state_after=state_after,
                vote_type=vote_type
            )
            
            if reward_components is None:
                return None
            
            # Get interaction data for RL tuple
            interaction_data = self.interaction_tracker.get_interaction_data(interaction_id)
            if not interaction_data:
                return None
            
            # Construct RL experience tuple
            state_before = interaction_data['state_before']
            reward = reward_components.total_reward
            done = False  # Episodes don't end with single questions
            
            # Action needs to be reconstructed from concept_id and difficulty
            # This would need to be stored in the interaction data or reconstructed
            action = 0  # Placeholder - needs proper implementation
            
            return (reward_components, state_before, action, reward, state_after, done)
            
        except Exception as e:
            logger.error(f"Error completing interaction: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reward system statistics"""
        pending = self.interaction_tracker.get_pending_interactions()
        completed = self.interaction_tracker.get_completed_interactions()
        
        # Calculate average rewards
        if completed:
            total_rewards = [data['reward_components'].total_reward for data in completed.values()]
            avg_reward = np.mean(total_rewards)
            reward_std = np.std(total_rewards)
        else:
            avg_reward = 0.0
            reward_std = 0.0
        
        return {
            'pending_interactions': len(pending),
            'completed_interactions': len(completed),
            'average_reward': avg_reward,
            'reward_std': reward_std,
            'reward_weights': {
                'vote': self.reward_calculator.w_vote,
                'learn': self.reward_calculator.w_learn,
                'effort': self.reward_calculator.w_effort
            }
        }
    
    def cleanup(self):
        """Cleanup old interactions"""
        self.interaction_tracker.cleanup_old_interactions()