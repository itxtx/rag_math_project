# src/rl_engine/train.py - Improved version with all fixes
import asyncio
import argparse
import logging
import os
import sys
import time
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rl_engine.rl_question_selector import RLQuestionSelector
from src.rl_engine.environment import RLEnvironment
from src.rl_engine.agent import RLAgent
from src.rl_engine.reward_system import RewardSystemManager, VoteType
from src.data_ingestion import vector_store_manager
from src.learner_model.profile_manager import LearnerProfileManager
from src.retrieval.retriever import HybridRetriever
from src.generation.question_generator_rag import RAGQuestionGenerator
from src import config

# Ensure logs directory exists
logs_dir = os.path.join(project_root, 'data', 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'rl_training.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ImprovedRLTrainingSimulator:
    """
    Enhanced training simulator with more realistic user behavior
    """
    
    def __init__(self, profile_manager):
        self.profile_manager = profile_manager
        
        # More sophisticated feedback patterns
        self.engagement_patterns = {
            'too_easy_penalty': 0.8,      # Higher penalty for too easy questions
            'appropriate_bonus': 0.9,     # High reward for appropriate questions
            'too_hard_tolerance': 0.4,    # Some tolerance for challenging questions
            'mastered_fatigue': 0.1,      # Very low engagement for mastered concepts
            'progression_bonus': 0.1,     # Bonus for good learning progression
            'streak_bonus': 0.05          # Small bonus for consecutive good questions
        }
        
        # Track learner states for more realistic simulation
        self.learner_states = {}
    
    def _get_learner_state(self, learner_id: str) -> Dict:
        """Get or initialize learner state for simulation"""
        if learner_id not in self.learner_states:
            self.learner_states[learner_id] = {
                'recent_questions': [],
                'engagement_history': [],
                'session_fatigue': 0.0,
                'learning_momentum': 0.5
            }
        return self.learner_states[learner_id]
    
    async def simulate_user_feedback(self, 
                                   learner_id: str,
                                   concept_id: str,
                                   difficulty: str,
                                   state,
                                   question_quality_hint: float = 0.5) -> VoteType:
        """
        Simulate realistic user feedback with improved behavioral modeling
        """
        learner_state = self._get_learner_state(learner_id)
        
        # Get learner's knowledge of the concept
        knowledge = self.profile_manager.get_concept_knowledge(learner_id, concept_id)
        concept_score = (knowledge.get('current_score', 0.0) / 10.0) if knowledge else 0.0
        attempts = knowledge.get('total_attempts', 0) if knowledge else 0
        
        # Determine question appropriateness
        difficulty_levels = {'easy': 0.3, 'medium': 0.6, 'hard': 0.9}
        question_difficulty = difficulty_levels.get(difficulty, 0.6)
        
        # Calculate base engagement probability
        if concept_score > 0.95:
            # Concept mastered - user likely tired of it
            base_prob = self.engagement_patterns['mastered_fatigue']
        elif abs(concept_score - question_difficulty) < 0.2:
            # Question difficulty matches learner level
            base_prob = self.engagement_patterns['appropriate_bonus']
        elif question_difficulty < concept_score - 0.3:
            # Question too easy
            base_prob = self.engagement_patterns['too_easy_penalty'] * (1.0 - concept_score)
        elif question_difficulty > concept_score + 0.4:
            # Question too hard
            base_prob = self.engagement_patterns['too_hard_tolerance']
        else:
            # Moderate mismatch
            base_prob = 0.6
        
        # Apply session-based modifiers
        session_data = state.session_data
        session_length = session_data.get('questions_answered', 0)
        avg_engagement = session_data.get('average_engagement', 0.5)
        momentum = session_data.get('momentum', 0.5)
        
        # Session fatigue factor
        fatigue_factor = max(0.2, 1.0 - (session_length * 0.04))  # Gradual fatigue
        
        # Momentum factor (recent performance trend)
        momentum_factor = 0.8 + (momentum * 0.4)  # Scale momentum impact
        
        # Learning progression bonus
        if attempts > 0 and concept_score > 0.1:
            # Reward questions that help learning progress
            progression_bonus = self.engagement_patterns['progression_bonus'] * concept_score
        else:
            progression_bonus = 0.0
        
        # Consecutive good questions streak bonus
        recent_engagement = learner_state['engagement_history'][-3:]
        if len(recent_engagement) >= 2 and all(e > 0.7 for e in recent_engagement):
            streak_bonus = self.engagement_patterns['streak_bonus']
        else:
            streak_bonus = 0.0
        
        # Combine all factors
        final_prob = (base_prob * fatigue_factor * momentum_factor + 
                     progression_bonus + streak_bonus + 
                     question_quality_hint * 0.1)  # Slight influence from question quality
        
        # Add some randomness for realism
        noise = np.random.normal(0, 0.1)
        final_prob = np.clip(final_prob + noise, 0.05, 0.95)
        
        # Update learner state
        learner_state['recent_questions'].append({
            'concept_id': concept_id,
            'difficulty': difficulty,
            'appropriateness': abs(concept_score - question_difficulty)
        })
        if len(learner_state['recent_questions']) > 5:
            learner_state['recent_questions'].pop(0)
        
        # Determine feedback
        if np.random.random() < final_prob:
            feedback = VoteType.UPVOTE
            engagement_score = min(final_prob + 0.1, 1.0)
        else:
            feedback = VoteType.DOWNVOTE
            engagement_score = max(final_prob - 0.2, 0.0)
        
        # Update engagement history
        learner_state['engagement_history'].append(engagement_score)
        if len(learner_state['engagement_history']) > 10:
            learner_state['engagement_history'].pop(0)
        
        # Update momentum based on recent trend
        if len(learner_state['engagement_history']) >= 3:
            recent_trend = np.mean(learner_state['engagement_history'][-3:])
            learner_state['learning_momentum'] = 0.7 * learner_state['learning_momentum'] + 0.3 * recent_trend
        
        logger.debug(f"Simulated feedback for learner {learner_id}: {feedback.value} "
                    f"(prob={final_prob:.3f}, concept_score={concept_score:.3f}, "
                    f"difficulty={question_difficulty:.3f})")
        
        return feedback

class ImprovedRLTrainer:
    """
    Enhanced RL trainer with better monitoring and validation
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.training_stats = {
            'episodes': 0,
            'total_interactions': 0,
            'average_reward': 0.0,
            'average_loss': 0.0,
            'upvotes': 0,
            'downvotes': 0,
            'training_start_time': None,
            'episode_rewards': [],
            'episode_losses': [],
            'validation_scores': [],
            'learning_curve_data': []
        }
        
        # Components
        self.rl_selector = None
        self.simulator = None
        self.learner_ids = []
        self.validation_learner_ids = []
        
        # Enhanced monitoring
        self.performance_monitor = {
            'best_average_reward': -float('inf'),
            'episodes_without_improvement': 0,
            'early_stopping_patience': 50,
            'convergence_threshold': 0.001
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration with better defaults"""
        default_config = {
            'training_episodes': 1000,
            'interactions_per_episode': 25,  # Slightly longer episodes
            'num_virtual_learners': 15,      # More learners for diversity
            'num_validation_learners': 5,   # Separate validation set
            'save_interval': 50,             # More frequent saves
            'evaluation_interval': 25,      # More frequent evaluation
            'max_training_time_hours': 12,   # Reasonable training time
            'target_average_reward': 0.6,    # Slightly higher target
            'model_save_path': 'models/rl/question_selector.pt',
            'plots_save_path': 'data/plots/rl_training.png',
            'early_stopping': True,
            'learning_rate_decay': True,
            'adaptive_epsilon': True
        }
        
        if config_path and os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    async def initialize(self):
        """Initialize training components with better error handling"""
        logger.info("Initializing enhanced RL training environment...")
        
        try:
            # Initialize Weaviate and components
            weaviate_client = vector_store_manager.get_weaviate_client()
            vector_store_manager.create_weaviate_schema(weaviate_client)
            
            profile_manager = LearnerProfileManager()
            retriever = HybridRetriever(weaviate_client=weaviate_client)
            question_generator = RAGQuestionGenerator()
            
            # Initialize RL selector with validation
            self.rl_selector = RLQuestionSelector(
                profile_manager=profile_manager,
                retriever=retriever,
                question_generator=question_generator,
                model_path=self.config['model_save_path']
            )
            
            await self.rl_selector.initialize()
            self.rl_selector.enable_training_mode()
            
            # Validate RL system is working
            if not self.rl_selector.is_initialized:
                raise RuntimeError("RL selector failed to initialize properly")
            
            # Initialize improved simulator
            self.simulator = ImprovedRLTrainingSimulator(profile_manager)
            
            # Create training and validation learners
            await self._create_virtual_learners()
            
            logger.info(f"Training environment initialized successfully:")
            logger.info(f"  Training learners: {len(self.learner_ids)}")
            logger.info(f"  Validation learners: {len(self.validation_learner_ids)}")
            logger.info(f"  Available concepts: {len(self.rl_selector.environment.available_concepts)}")
            logger.info(f"  Action space size: {self.rl_selector.environment.action_space_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize training environment: {e}")
            raise
    
    async def _create_virtual_learners(self):
        """Create virtual learners with more realistic diversity"""
        # Training learners
        self.learner_ids = []
        for i in range(self.config['num_virtual_learners']):
            learner_id = f"train_learner_{i:03d}"
            self.learner_ids.append(learner_id)
            await self.rl_selector.profile_manager.create_profile(learner_id)
            
            # Create diverse learning backgrounds
            background_type = i % 4  # 4 different background types
            await self._simulate_learner_background(learner_id, background_type)
        
        # Validation learners (separate from training)
        self.validation_learner_ids = []
        for i in range(self.config['num_validation_learners']):
            learner_id = f"val_learner_{i:03d}"
            self.validation_learner_ids.append(learner_id)
            await self.rl_selector.profile_manager.create_profile(learner_id)
            
            # Create different backgrounds for validation
            background_type = (i + 2) % 4  # Offset to ensure different patterns
            await self._simulate_learner_background(learner_id, background_type)
        
        logger.info(f"Created {len(self.learner_ids)} training and {len(self.validation_learner_ids)} validation learners")
    
    async def _simulate_learner_background(self, learner_id: str, background_type: int):
        """Simulate diverse learner backgrounds"""
        concepts = self.rl_selector.environment.available_concepts
        if not concepts:
            return
        
        # Different background patterns
        if background_type == 0:
            # Beginner: little prior knowledge
            num_concepts = min(2, len(concepts))
            score_range = (1.0, 4.0)
        elif background_type == 1:
            # Intermediate: moderate knowledge
            num_concepts = min(5, len(concepts))
            score_range = (3.0, 7.0)
        elif background_type == 2:
            # Advanced: good knowledge but some gaps
            num_concepts = min(8, len(concepts))
            score_range = (5.0, 9.0)
        else:
            # Mixed: very uneven knowledge
            num_concepts = min(6, len(concepts))
            score_range = (1.0, 9.0)
        
        if num_concepts > 0:
            selected_concepts = np.random.choice(concepts, size=num_concepts, replace=False)
            
            for concept_id in selected_concepts:
                # Simulate learning history
                num_attempts = np.random.randint(1, 6)
                
                for attempt in range(num_attempts):
                    if background_type == 3:  # Mixed background - more variance
                        score = np.random.uniform(score_range[0], score_range[1])
                    else:
                        # Gradual improvement
                        base_score = np.random.uniform(score_range[0], score_range[1])
                        improvement = attempt * np.random.uniform(0.2, 0.8)
                        score = min(10.0, base_score + improvement)
                    
                    answered_correctly = score >= 6.0
                    
                    # Update knowledge with realistic SRS data
                    srs_details = {
                        'next_interval_days': max(1, attempt + 1),
                        'next_review_at': datetime.now() + timedelta(days=attempt + 1),
                        'new_srs_repetitions': attempt if answered_correctly else 0
                    }
                    
                    success = self.rl_selector.profile_manager.update_concept_srs_and_difficulty(
                        learner_id=learner_id,
                        concept_id=concept_id,
                        doc_id="background_sim",
                        score=score,
                        answered_correctly=answered_correctly,
                        srs_details=srs_details
                    )
                    
                    if not success:
                        logger.warning(f"Failed to update background for {learner_id}, concept {concept_id}")
    
    async def run_training(self):
        """Run the enhanced training loop with better monitoring"""
        logger.info("Starting enhanced RL training...")
        self.training_stats['training_start_time'] = time.time()
        
        try:
            for episode in range(self.config['training_episodes']):
                episode_reward = await self._run_episode(episode)
                
                self.training_stats['episodes'] = episode + 1
                self.training_stats['episode_rewards'].append(episode_reward)
                
                # Update running averages
                recent_rewards = self.training_stats['episode_rewards'][-50:]  # Last 50 episodes
                self.training_stats['average_reward'] = np.mean(recent_rewards)
                
                # Adaptive learning rate decay
                if self.config.get('learning_rate_decay') and episode > 0 and episode % 100 == 0:
                    self._apply_learning_rate_decay()
                
                # Adaptive epsilon decay
                if self.config.get('adaptive_epsilon'):
                    self._apply_adaptive_epsilon(episode)
                
                # Enhanced logging
                if (episode + 1) % 10 == 0:
                    self._log_training_progress(episode, episode_reward)
                
                # Save model periodically
                if (episode + 1) % self.config['save_interval'] == 0:
                    self.rl_selector.save_model()
                    logger.info(f"Model saved at episode {episode + 1}")
                
                # Enhanced evaluation with validation
                if (episode + 1) % self.config['evaluation_interval'] == 0:
                    validation_score = await self._evaluate_model(episode + 1)
                    self.training_stats['validation_scores'].append(validation_score)
                    
                    # Check for improvement and early stopping
                    if self._check_early_stopping(validation_score):
                        logger.info(f"Early stopping triggered at episode {episode + 1}")
                        break
                
                # Check other stopping conditions
                if self._should_stop_training():
                    logger.info("Training stopping criteria met")
                    break
            
            # Final evaluation and model save
            self.rl_selector.save_model()
            final_score = await self._evaluate_model(self.training_stats['episodes'], final=True)
            
            # Generate comprehensive plots
            self._generate_training_plots()
            
            logger.info("Enhanced RL training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _apply_learning_rate_decay(self):
        """Apply learning rate decay"""
        if hasattr(self.rl_selector.agent, 'optimizer'):
            current_lr = self.rl_selector.agent.optimizer.param_groups[0]['lr']
            new_lr = current_lr * 0.95  # 5% decay
            self.rl_selector.agent.optimizer.param_groups[0]['lr'] = new_lr
            logger.info(f"Learning rate decayed to {new_lr:.6f}")
    
    def _apply_adaptive_epsilon(self, episode: int):
        """Apply adaptive epsilon based on performance"""
        if len(self.training_stats['episode_rewards']) >= 20:
            recent_performance = np.mean(self.training_stats['episode_rewards'][-20:])
            
            # If performance is poor, increase exploration
            if recent_performance < 0.3:
                self.rl_selector.agent.epsilon = min(0.3, self.rl_selector.agent.epsilon * 1.05)
            # If performance is good, decrease exploration
            elif recent_performance > 0.7:
                self.rl_selector.agent.epsilon = max(0.01, self.rl_selector.agent.epsilon * 0.98)
    
    def _log_training_progress(self, episode: int, episode_reward: float):
        """Enhanced logging of training progress"""
        epsilon = self.rl_selector.agent.epsilon
        buffer_size = len(self.rl_selector.agent.replay_buffer)
        
        recent_losses = self.training_stats['episode_losses'][-10:]
        avg_loss = np.mean(recent_losses) if recent_losses else 0.0
        
        upvote_rate = (self.training_stats['upvotes'] / 
                      max(1, self.training_stats['upvotes'] + self.training_stats['downvotes']))
        
        logger.info(f"Episode {episode + 1}/{self.config['training_episodes']}: "
                   f"Reward={episode_reward:.3f}, "
                   f"Avg100={self.training_stats['average_reward']:.3f}, "
                   f"Epsilon={epsilon:.3f}, "
                   f"Loss={avg_loss:.4f}, "
                   f"Upvote%={upvote_rate:.1%}, "
                   f"Buffer={buffer_size}")
    
    def _check_early_stopping(self, validation_score: float) -> bool:
        """Check early stopping criteria"""
        if not self.config.get('early_stopping'):
            return False
        
        # Check for improvement
        if validation_score > self.performance_monitor['best_average_reward']:
            self.performance_monitor['best_average_reward'] = validation_score
            self.performance_monitor['episodes_without_improvement'] = 0
            return False
        else:
            self.performance_monitor['episodes_without_improvement'] += 1
            
            # Early stopping if no improvement for too long
            if (self.performance_monitor['episodes_without_improvement'] >= 
                self.performance_monitor['early_stopping_patience']):
                return True
        
        return False
    
    async def _run_episode(self, episode_num: int) -> float:
        """Run a single enhanced training episode"""
        total_episode_reward = 0.0
        episode_losses = []
        
        # Select multiple learners for this episode for diversity
        episode_learners = np.random.choice(self.learner_ids, size=min(3, len(self.learner_ids)), replace=False)
        
        interactions_per_learner = self.config['interactions_per_episode'] // len(episode_learners)
        
        for learner_id in episode_learners:
            for interaction in range(interactions_per_learner):
                try:
                    # Get next question from RL agent
                    question_data = await self.rl_selector.select_next_question(learner_id)
                    
                    if not question_data or "error" in question_data:
                        continue
                    
                    interaction_id = question_data.get("interaction_id")
                    if not interaction_id:
                        continue
                    
                    # Simulate learner answering the question
                    answer_quality = await self._simulate_answer_submission(learner_id, question_data)
                    
                    # Get current state for feedback simulation
                    current_state = await self.rl_selector.environment.get_current_state(learner_id)
                    
                    # Simulate user feedback with quality hint
                    feedback = await self.simulator.simulate_user_feedback(
                        learner_id=learner_id,
                        concept_id=question_data["concept_id"],
                        difficulty=question_data.get("difficulty", "medium"),
                        state=current_state,
                        question_quality_hint=answer_quality
                    )
                    
                    # Process feedback and get reward
                    success = await self.rl_selector.process_feedback(
                        interaction_id=interaction_id,
                        learner_id=learner_id,
                        concept_id=question_data["concept_id"],
                        vote_type=feedback
                    )
                    
                    if success:
                        # Get reward from completed interaction
                        interaction_data = self.rl_selector.reward_manager.interaction_tracker.get_interaction_data(interaction_id)
                        if interaction_data and interaction_data.get('completed'):
                            reward = interaction_data['reward_components'].total_reward
                            total_episode_reward += reward
                            
                            # Track feedback stats
                            if feedback == VoteType.UPVOTE:
                                self.training_stats['upvotes'] += 1
                            else:
                                self.training_stats['downvotes'] += 1
                    
                    # Train the agent
                    if self.rl_selector.agent.replay_buffer.can_sample(self.rl_selector.agent.batch_size):
                        loss = self.rl_selector.agent.train()
                        if loss is not None:
                            episode_losses.append(loss)
                    
                    self.training_stats['total_interactions'] += 1
                    
                except Exception as e:
                    logger.error(f"Error in episode {episode_num}, interaction {interaction}: {e}")
                    continue
        
        # Update average loss
        if episode_losses:
            avg_episode_loss = np.mean(episode_losses)
            self.training_stats['episode_losses'].append(avg_episode_loss)
            self.training_stats['average_loss'] = np.mean(self.training_stats['episode_losses'][-100:])
        
        return total_episode_reward
    
    async def _simulate_answer_submission(self, learner_id: str, question_data: Dict) -> float:
        """Simulate answer submission with quality feedback"""
        concept_id = question_data["concept_id"]
        
        # Get current knowledge
        knowledge = self.rl_selector.profile_manager.get_concept_knowledge(learner_id, concept_id)
        concept_score = (knowledge.get('current_score', 0.0) / 10.0) if knowledge else 0.0
        
        # Simulate answer quality based on knowledge and question appropriateness
        difficulty_levels = {'easy': 0.3, 'medium': 0.6, 'hard': 0.9}
        question_difficulty = difficulty_levels.get(question_data.get("difficulty", "medium"), 0.6)
        
        # More sophisticated answer simulation
        knowledge_factor = concept_score
        difficulty_match = 1.0 - abs(concept_score - question_difficulty)
        
        # Base accuracy probability
        if concept_score >= question_difficulty:
            base_accuracy = 0.7 + (difficulty_match * 0.2)
        else:
            base_accuracy = 0.2 + (concept_score / question_difficulty) * 0.5
        
        # Add learning effect (slight improvement over time)
        attempts = knowledge.get('total_attempts', 0) if knowledge else 0
        learning_bonus = min(0.1, attempts * 0.01)
        
        final_accuracy = np.clip(base_accuracy + learning_bonus, 0.05, 0.95)
        simulated_correct = np.random.random() < final_accuracy
        
        # Generate realistic score
        if simulated_correct:
            score_base = 0.6 + (final_accuracy * 0.4)
        else:
            score_base = 0.1 + (final_accuracy * 0.4)
        
        # Add noise for realism
        noise = np.random.normal(0, 0.1)
        final_score = np.clip(score_base + noise, 0.0, 1.0)
        
        # Update learner's knowledge
        graded_score = final_score * 10.0
        answered_correctly = final_score >= 0.7
        
        # Realistic SRS update
        current_reps = knowledge.get('srs_repetitions', 0) if knowledge else 0
        current_interval = knowledge.get('srs_interval_days', 0) if knowledge else 0
        
        if answered_correctly:
            new_reps = current_reps + 1
            new_interval = max(1, int(current_interval * 1.5)) if current_interval > 0 else 1
        else:
            new_reps = 0
            new_interval = 1
        
        srs_details = {
            'next_interval_days': new_interval,
            'next_review_at': datetime.now() + timedelta(days=new_interval),
            'new_srs_repetitions': new_reps
        }
        
        success = self.rl_selector.profile_manager.update_concept_srs_and_difficulty(
            learner_id=learner_id,
            concept_id=concept_id,
            doc_id=question_data.get("doc_id", "training_doc"),
            score=graded_score,
            answered_correctly=answered_correctly,
            srs_details=srs_details
        )
        
        if not success:
            logger.warning(f"Failed to update knowledge for {learner_id}, concept {concept_id}")
        
        # Return quality score for feedback simulation
        return difficulty_match * 0.7 + final_accuracy * 0.3
    
    async def _evaluate_model(self, episode: int, final: bool = False) -> float:
        """Enhanced model evaluation with validation set"""
        logger.info(f"Evaluating model at episode {episode}...")
        
        # Temporarily disable training mode
        original_epsilon = self.rl_selector.agent.epsilon
        self.rl_selector.agent.epsilon = 0.01  # Minimal exploration
        
        try:
            evaluation_rewards = []
            evaluation_feedback = {'up': 0, 'down': 0}
            evaluation_accuracy = []
            
            # Use validation learners for unbiased evaluation
            for learner_id in self.validation_learner_ids:
                episode_reward = 0.0
                episode_accuracy = []
                
                for _ in range(10):  # More questions per evaluation
                    try:
                        question_data = await self.rl_selector.select_next_question(learner_id)
                        
                        if not question_data or "error" in question_data:
                            continue
                        
                        interaction_id = question_data.get("interaction_id")
                        if not interaction_id:
                            continue
                        
                        # Simulate answer
                        answer_quality = await self._simulate_answer_submission(learner_id, question_data)
                        episode_accuracy.append(answer_quality)
                        
                        # Get state for feedback
                        current_state = await self.rl_selector.environment.get_current_state(learner_id)
                        
                        # Simulate feedback
                        feedback = await self.simulator.simulate_user_feedback(
                            learner_id=learner_id,
                            concept_id=question_data["concept_id"],
                            difficulty=question_data.get("difficulty", "medium"),
                            state=current_state,
                            question_quality_hint=answer_quality
                        )
                        
                        # Process feedback
                        success = await self.rl_selector.process_feedback(
                            interaction_id=interaction_id,
                            learner_id=learner_id,
                            concept_id=question_data["concept_id"],
                            vote_type=feedback
                        )
                        
                        if success:
                            interaction_data = self.rl_selector.reward_manager.interaction_tracker.get_interaction_data(interaction_id)
                            if interaction_data and interaction_data.get('completed'):
                                reward = interaction_data['reward_components'].total_reward
                                episode_reward += reward
                                
                                if feedback == VoteType.UPVOTE:
                                    evaluation_feedback['up'] += 1
                                else:
                                    evaluation_feedback['down'] += 1
                    
                    except Exception as e:
                        logger.error(f"Error in evaluation: {e}")
                        continue
                
                evaluation_rewards.append(episode_reward)
                if episode_accuracy:
                    evaluation_accuracy.extend(episode_accuracy)
            
            # Calculate metrics
            avg_reward = np.mean(evaluation_rewards) if evaluation_rewards else 0.0
            avg_accuracy = np.mean(evaluation_accuracy) if evaluation_accuracy else 0.0
            total_feedback = evaluation_feedback['up'] + evaluation_feedback['down']
            upvote_rate = evaluation_feedback['up'] / total_feedback if total_feedback > 0 else 0.0
            
            # Log results
            logger.info(f"Evaluation Results (Episode {episode}):")
            logger.info(f"  Average Reward: {avg_reward:.3f}")
            logger.info(f"  Average Accuracy: {avg_accuracy:.3f}")
            logger.info(f"  Upvote Rate: {upvote_rate:.2%}")
            logger.info(f"  Total Evaluations: {len(evaluation_rewards)}")
            
            if final:
                logger.info("=== FINAL EVALUATION ===")
                logger.info(f"Total Episodes: {self.training_stats['episodes']}")
                logger.info(f"Total Interactions: {self.training_stats['total_interactions']}")
                logger.info(f"Final Average Reward: {avg_reward:.3f}")
                logger.info(f"Final Upvote Rate: {upvote_rate:.2%}")
                logger.info(f"Training Time: {(time.time() - self.training_stats['training_start_time']) / 3600:.2f} hours")
            
            return avg_reward
        
        finally:
            # Restore original epsilon
            self.rl_selector.agent.epsilon = original_epsilon
    
    def _should_stop_training(self) -> bool:
        """Enhanced stopping criteria"""
        # Time limit
        if self.training_stats['training_start_time']:
            elapsed_hours = (time.time() - self.training_stats['training_start_time']) / 3600
            if elapsed_hours >= self.config['max_training_time_hours']:
                logger.info(f"Training time limit reached: {elapsed_hours:.2f} hours")
                return True
        
        # Reward convergence
        if len(self.training_stats['episode_rewards']) >= 100:
            recent_rewards = self.training_stats['episode_rewards'][-100:]
            if np.mean(recent_rewards) >= self.config['target_average_reward']:
                logger.info(f"Target average reward reached: {np.mean(recent_rewards):.3f}")
                return True
        
        # Loss convergence (check for plateau)
        if len(self.training_stats['episode_losses']) >= 50:
            recent_losses = self.training_stats['episode_losses'][-50:]
            loss_std = np.std(recent_losses)
            if loss_std < self.performance_monitor['convergence_threshold']:
                logger.info(f"Loss converged (std={loss_std:.4f})")
                return True
        
        return False
    
    def _generate_training_plots(self):
        """Generate comprehensive training plots"""
        try:
            fig = plt.figure(figsize=(20, 12))
            
            # Create a 3x3 grid
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Episode rewards with trend
            ax1 = fig.add_subplot(gs[0, 0])
            if self.training_stats['episode_rewards']:
                episodes = range(1, len(self.training_stats['episode_rewards']) + 1)
                ax1.plot(episodes, self.training_stats['episode_rewards'], alpha=0.3, color='blue', label='Episode Rewards')
                
                # Moving averages
                if len(self.training_stats['episode_rewards']) >= 20:
                    ma20 = self._moving_average(self.training_stats['episode_rewards'], 20)
                    ax1.plot(range(20, len(self.training_stats['episode_rewards']) + 1), ma20, 
                            color='red', linewidth=2, label='MA(20)')
                
                if len(self.training_stats['episode_rewards']) >= 50:
                    ma50 = self._moving_average(self.training_stats['episode_rewards'], 50)
                    ax1.plot(range(50, len(self.training_stats['episode_rewards']) + 1), ma50, 
                            color='green', linewidth=2, label='MA(50)')
                
                ax1.set_title('Episode Rewards Over Time')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Total Reward')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # 2. Training losses
            ax2 = fig.add_subplot(gs[0, 1])
            if self.training_stats['episode_losses']:
                episodes = range(1, len(self.training_stats['episode_losses']) + 1)
                ax2.plot(episodes, self.training_stats['episode_losses'], color='orange', alpha=0.7)
                
                if len(self.training_stats['episode_losses']) >= 10:
                    ma10 = self._moving_average(self.training_stats['episode_losses'], 10)
                    ax2.plot(range(10, len(self.training_stats['episode_losses']) + 1), ma10, 
                            color='red', linewidth=2, label='MA(10)')
                
                ax2.set_title('Training Loss')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Loss')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # 3. Validation scores
            ax3 = fig.add_subplot(gs[0, 2])
            if self.training_stats['validation_scores']:
                eval_episodes = range(self.config['evaluation_interval'], 
                                    len(self.training_stats['validation_scores']) * self.config['evaluation_interval'] + 1,
                                    self.config['evaluation_interval'])
                ax3.plot(eval_episodes, self.training_stats['validation_scores'], 'g-o', linewidth=2, markersize=4)
                ax3.set_title('Validation Performance')
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('Validation Score')
                ax3.grid(True, alpha=0.3)
            
            # 4. Feedback distribution
            ax4 = fig.add_subplot(gs[1, 0])
            total_up = self.training_stats['upvotes']
            total_down = self.training_stats['downvotes']
            if total_up + total_down > 0:
                labels = ['Upvotes', 'Downvotes']
                sizes = [total_up, total_down]
                colors = ['#2ecc71', '#e74c3c']
                
                wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, 
                                                 autopct='%1.1f%%', startangle=90)
                ax4.set_title('User Feedback Distribution')
                
                # Add count annotations
                total_feedback = total_up + total_down
                ax4.text(0, -1.3, f'Total: {total_feedback}', ha='center', transform=ax4.transAxes)
            
            # 5. Reward distribution histogram
            ax5 = fig.add_subplot(gs[1, 1])
            if self.training_stats['episode_rewards']:
                ax5.hist(self.training_stats['episode_rewards'], bins=30, alpha=0.7, 
                        color='skyblue', edgecolor='black')
                mean_reward = np.mean(self.training_stats['episode_rewards'])
                median_reward = np.median(self.training_stats['episode_rewards'])
                ax5.axvline(mean_reward, color='red', linestyle='--', 
                           label=f'Mean: {mean_reward:.3f}')
                ax5.axvline(median_reward, color='green', linestyle='--',
                           label=f'Median: {median_reward:.3f}')
                ax5.set_title('Reward Distribution')
                ax5.set_xlabel('Episode Reward')
                ax5.set_ylabel('Frequency')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
            
            # 6. Learning curve comparison
            ax6 = fig.add_subplot(gs[1, 2])
            if len(self.training_stats['episode_rewards']) > 100:
                # Split into phases for comparison
                rewards = self.training_stats['episode_rewards']
                phase_size = len(rewards) // 3
                
                early = rewards[:phase_size]
                mid = rewards[phase_size:2*phase_size]
                late = rewards[2*phase_size:]
                
                phases = [early, mid, late]
                labels = ['Early', 'Mid', 'Late']
                colors = ['lightcoral', 'lightblue', 'lightgreen']
                
                box_plot = ax6.boxplot(phases, labels=labels, patch_artist=True)
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax6.set_title('Learning Progress by Phase')
                ax6.set_ylabel('Episode Reward')
                ax6.grid(True, alpha=0.3)
            
            # 7. Agent statistics text
            ax7 = fig.add_subplot(gs[2, 0])
            if hasattr(self.rl_selector, 'agent') and self.rl_selector.agent:
                agent_stats = self.rl_selector.agent.get_stats()
                
                # Calculate additional metrics
                upvote_rate = total_up / max(1, total_up + total_down)
                training_time = (time.time() - self.training_stats['training_start_time']) / 3600
                
                stats_text = f"""Training Statistics:
Episodes: {self.training_stats['episodes']}
Total Interactions: {self.training_stats['total_interactions']}
Training Time: {training_time:.2f}h
Final Epsilon: {agent_stats['epsilon']:.4f}
Buffer Size: {agent_stats['buffer_size']}
Upvote Rate: {upvote_rate:.2%}
Avg Reward: {self.training_stats['average_reward']:.3f}
Avg Loss: {self.training_stats['average_loss']:.4f}
Device: {agent_stats['device']}"""
                
                ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=10,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
                ax7.set_title('Training Summary')
                ax7.axis('off')
            
            # 8. Hyperparameters and config
            ax8 = fig.add_subplot(gs[2, 1])
            config_text = f"""Configuration:
Training Episodes: {self.config['training_episodes']}
Interactions/Episode: {self.config['interactions_per_episode']}
Virtual Learners: {self.config['num_virtual_learners']}
Validation Learners: {self.config['num_validation_learners']}
Learning Rate: {self.rl_selector.agent.learning_rate:.6f}
Gamma: {self.rl_selector.agent.gamma:.3f}
Batch Size: {self.rl_selector.agent.batch_size}
Target Update: {self.rl_selector.agent.target_update_freq}
Buffer Size: {self.rl_selector.agent.replay_buffer.capacity}"""
            
            ax8.text(0.05, 0.95, config_text, transform=ax8.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
            ax8.set_title('Hyperparameters')
            ax8.axis('off')
            
            # 9. Performance metrics over time
            ax9 = fig.add_subplot(gs[2, 2])
            if len(self.training_stats['episode_rewards']) >= 50:
                # Calculate rolling performance metrics
                window = 25
                rolling_mean = self._moving_average(self.training_stats['episode_rewards'], window)
                rolling_std = self._moving_std(self.training_stats['episode_rewards'], window)
                
                episodes = range(window, len(self.training_stats['episode_rewards']) + 1)
                
                ax9.plot(episodes, rolling_mean, 'b-', label=f'Mean (MA{window})', linewidth=2)
                ax9.fill_between(episodes, 
                               np.array(rolling_mean) - np.array(rolling_std),
                               np.array(rolling_mean) + np.array(rolling_std),
                               alpha=0.3, color='blue', label='±1 Std')
                
                ax9.set_title('Performance Stability')
                ax9.set_xlabel('Episode')
                ax9.set_ylabel('Reward')
                ax9.legend()
                ax9.grid(True, alpha=0.3)
            
            plt.suptitle(f'RL Training Results - {self.training_stats["episodes"]} Episodes', fontsize=16)
            
            # Save plot
            os.makedirs(os.path.dirname(self.config['plots_save_path']), exist_ok=True)
            plt.savefig(self.config['plots_save_path'], dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Comprehensive training plots saved to {self.config['plots_save_path']}")
            
        except Exception as e:
            logger.error(f"Error generating training plots: {e}")
    
    def _moving_average(self, data: List[float], window: int) -> List[float]:
        """Calculate moving average"""
        return [np.mean(data[i-window+1:i+1]) for i in range(window-1, len(data))]
    
    def _moving_std(self, data: List[float], window: int) -> List[float]:
        """Calculate moving standard deviation"""
        return [np.std(data[i-window+1:i+1]) for i in range(window-1, len(data))]
    
    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary"""
        training_time = (time.time() - self.training_stats['training_start_time']) / 3600 if self.training_stats['training_start_time'] else 0
        total_feedback = self.training_stats['upvotes'] + self.training_stats['downvotes']
        
        return {
            'episodes_completed': self.training_stats['episodes'],
            'total_interactions': self.training_stats['total_interactions'],
            'training_time_hours': training_time,
            'final_average_reward': self.training_stats['average_reward'],
            'final_average_loss': self.training_stats['average_loss'],
            'upvotes': self.training_stats['upvotes'],
            'downvotes': self.training_stats['downvotes'],
            'upvote_rate': self.training_stats['upvotes'] / max(1, total_feedback),
            'best_validation_score': max(self.training_stats['validation_scores']) if self.training_stats['validation_scores'] else 0.0,
            'reward_improvement': (self.training_stats['episode_rewards'][-1] - self.training_stats['episode_rewards'][0]) if len(self.training_stats['episode_rewards']) > 1 else 0.0,
            'convergence_achieved': self.training_stats['average_reward'] >= self.config['target_average_reward'],
            'agent_stats': self.rl_selector.agent.get_stats() if self.rl_selector and self.rl_selector.agent else {},
            'rl_system_stats': self.rl_selector.get_stats() if self.rl_selector else {}
        }

async def main():
    """Enhanced main training function"""
    parser = argparse.ArgumentParser(description='Train Enhanced RL Question Selector')
    parser.add_argument('--config', type=str, help='Path to training configuration file')
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--learners', type=int, default=15, help='Number of virtual learners')
    parser.add_argument('--interactions', type=int, default=25, help='Interactions per episode')
    parser.add_argument('--save-interval', type=int, default=50, help='Model save interval')
    parser.add_argument('--eval-interval', type=int, default=25, help='Evaluation interval')
    parser.add_argument('--model-path', type=str, default='models/rl/question_selector.pt', help='Model save path')
    parser.add_argument('--quick', action='store_true', help='Quick training with reduced episodes')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create trainer
    trainer = ImprovedRLTrainer(args.config)
    
    # Override config with command line arguments
    if args.episodes:
        trainer.config['training_episodes'] = args.episodes
    if args.learners:
        trainer.config['num_virtual_learners'] = args.learners
    if args.interactions:
        trainer.config['interactions_per_episode'] = args.interactions
    if args.save_interval:
        trainer.config['save_interval'] = args.save_interval
    if args.eval_interval:
        trainer.config['evaluation_interval'] = args.eval_interval
    if args.model_path:
        trainer.config['model_save_path'] = args.model_path
    
    # Quick training mode
    if args.quick:
        trainer.config.update({
            'training_episodes': 100,
            'num_virtual_learners': 8,
            'interactions_per_episode': 15,
            'save_interval': 25,
            'evaluation_interval': 20
        })
        logger.info("Quick training mode enabled")
    
    # Ensure directories exist
    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('data/plots', exist_ok=True)
    os.makedirs(os.path.dirname(trainer.config['model_save_path']), exist_ok=True)
    
    try:
        logger.info("=== Enhanced RL Training Starting ===")
        logger.info(f"Configuration: {trainer.config}")
        
        # Initialize and run training
        await trainer.initialize()
        await trainer.run_training()
        
        # Print comprehensive summary
        summary = trainer.get_training_summary()
        logger.info("=== ENHANCED TRAINING SUMMARY ===")
        for key, value in summary.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            elif isinstance(value, dict):
                logger.info(f"{key}: {len(value)} entries")
            else:
                logger.info(f"{key}: {value}")
        
        # Convergence analysis
        if summary['convergence_achieved']:
            logger.info("🎉 Training converged successfully!")
        else:
            logger.info("⚠️  Training completed but target performance not reached")
            logger.info(f"   Target: {trainer.config['target_average_reward']:.3f}")
            logger.info(f"   Achieved: {summary['final_average_reward']:.3f}")
        
        # Final recommendations
        if summary['upvote_rate'] > 0.7:
            logger.info("✅ Good user engagement achieved")
        else:
            logger.info("⚠️  Consider tuning reward function for better engagement")
        
        logger.info("Enhanced RL training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if trainer.rl_selector:
            trainer.rl_selector.save_model()
            logger.info("Model saved before exit")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
