# src/rl_engine/train.py
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/rl_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RLTrainingSimulator:
    """
    Simulates user interactions for RL training
    Generates synthetic feedback based on question quality heuristics
    """
    
    def __init__(self, profile_manager):
        self.profile_manager = profile_manager
        self.feedback_patterns = {
            'too_easy': 0.2,      # Probability of downvote if question is too easy
            'appropriate': 0.8,   # Probability of upvote if question is appropriate
            'too_hard': 0.3,      # Probability of upvote if question is too hard
            'mastered': 0.1       # Probability of upvote if concept already mastered
        }
    
    async def simulate_user_feedback(self, 
                                   learner_id: str,
                                   concept_id: str,
                                   difficulty: str,
                                   state) -> VoteType:
        """
        Simulate realistic user feedback based on question appropriateness
        """
        # Get learner's knowledge of the concept
        knowledge = await self.profile_manager.get_concept_knowledge(learner_id, concept_id)
        concept_score = (knowledge.get('current_score', 0.0) / 10.0) if knowledge else 0.0
        attempts = knowledge.get('total_attempts', 0) if knowledge else 0
        
        # Determine question appropriateness
        difficulty_levels = {'easy': 0.3, 'medium': 0.6, 'hard': 0.9}
        question_difficulty = difficulty_levels.get(difficulty, 0.6)
        
        # Heuristics for feedback simulation
        if concept_score > 0.9:
            # Concept already mastered - user likely finds continued questions annoying
            feedback_prob = self.feedback_patterns['mastered']
        elif abs(concept_score - question_difficulty) < 0.2:
            # Question difficulty matches learner level - appropriate
            feedback_prob = self.feedback_patterns['appropriate']
        elif question_difficulty < concept_score - 0.3:
            # Question too easy
            feedback_prob = self.feedback_patterns['too_easy']
        elif question_difficulty > concept_score + 0.3:
            # Question too hard  
            feedback_prob = self.feedback_patterns['too_hard']
        else:
            # Neutral case
            feedback_prob = 0.5
        
        # Add some randomness and session fatigue
        session_length = state.session_length
        fatigue_factor = max(0.1, 1.0 - (session_length * 0.05))  # Reduce positive feedback with session length
        
        final_prob = feedback_prob * fatigue_factor
        
        # Generate feedback
        if np.random.random() < final_prob:
            return VoteType.UPVOTE
        else:
            return VoteType.DOWNVOTE

class RLTrainer:
    """
    Main RL training orchestrator
    Manages training episodes, data collection, and model updates
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
            'episode_losses': []
        }
        
        # Components
        self.rl_selector = None
        self.simulator = None
        self.learner_ids = []
    
    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration"""
        default_config = {
            'training_episodes': 1000,
            'interactions_per_episode': 20,
            'num_virtual_learners': 10,
            'save_interval': 100,
            'evaluation_interval': 50,
            'max_training_time_hours': 24,
            'target_average_reward': 0.5,
            'model_save_path': 'models/rl/question_selector.pt',
            'plots_save_path': 'data/plots/rl_training.png'
        }
        
        if config_path and os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    async def initialize(self):
        """Initialize training components"""
        logger.info("Initializing RL training environment...")
        
        try:
            # Initialize Weaviate and components
            weaviate_client = vector_store_manager.get_weaviate_client()
            vector_store_manager.create_weaviate_schema(weaviate_client)
            
            profile_manager = LearnerProfileManager()
            retriever = HybridRetriever(weaviate_client=weaviate_client)
            question_generator = RAGQuestionGenerator()
            
            # Initialize RL selector
            self.rl_selector = RLQuestionSelector(
                profile_manager=profile_manager,
                retriever=retriever,
                question_generator=question_generator,
                model_path=self.config['model_save_path']
            )
            
            await self.rl_selector.initialize()
            self.rl_selector.enable_training_mode()
            
            # Initialize simulator
            self.simulator = RLTrainingSimulator(profile_manager)
            
            # Create virtual learners
            await self._create_virtual_learners()
            
            logger.info(f"Training environment initialized with {len(self.learner_ids)} virtual learners")
            
        except Exception as e:
            logger.error(f"Failed to initialize training environment: {e}")
            raise
    
    async def _create_virtual_learners(self):
        """Create virtual learners with diverse knowledge states"""
        self.learner_ids = []
        
        for i in range(self.config['num_virtual_learners']):
            learner_id = f"virtual_learner_{i:03d}"
            self.learner_ids.append(learner_id)
            
            # Create learner profile
            await self.rl_selector.profile_manager.create_profile(learner_id)
            
            # Simulate some prior learning history
            await self._simulate_prior_learning(learner_id)
        
        logger.info(f"Created {len(self.learner_ids)} virtual learners")
    
    async def _simulate_prior_learning(self, learner_id: str):
        """Simulate prior learning history for a virtual learner"""
        # Get available concepts
        concepts = self.rl_selector.environment.available_concepts
        if not concepts:
            return
        
        # Randomly assign some prior knowledge
        num_prior_concepts = np.random.randint(0, min(5, len(concepts)))
        prior_concepts = np.random.choice(concepts, size=num_prior_concepts, replace=False)
        
        for concept_id in prior_concepts:
            # Simulate some learning attempts
            num_attempts = np.random.randint(1, 8)
            for attempt in range(num_attempts):
                # Simulate improving scores over time
                base_score = np.random.uniform(2.0, 8.0)
                improvement = attempt * np.random.uniform(0.1, 0.5)
                score = min(10.0, base_score + improvement)
                
                # Update knowledge
                await self.rl_selector.profile_manager.update_concept_srs_and_difficulty(
                    learner_id=learner_id,
                    concept_id=concept_id,
                    doc_id="training_doc",
                    score=score,
                    answered_correctly=score >= 6.0,
                    srs_details={
                        'next_interval_days': 1,
                        'next_review_at': datetime.now(),
                        'new_srs_repetitions': attempt
                    }
                )
    
    async def run_training(self):
        """Run the main training loop"""
        logger.info("Starting RL training...")
        self.training_stats['training_start_time'] = time.time()
        
        try:
            for episode in range(self.config['training_episodes']):
                episode_reward = await self._run_episode(episode)
                
                self.training_stats['episodes'] = episode + 1
                self.training_stats['episode_rewards'].append(episode_reward)
                
                # Compute running averages
                recent_rewards = self.training_stats['episode_rewards'][-100:]
                self.training_stats['average_reward'] = np.mean(recent_rewards)
                
                # Log progress
                if (episode + 1) % 10 == 0:
                    logger.info(f"Episode {episode + 1}/{self.config['training_episodes']}: "
                               f"Reward={episode_reward:.3f}, "
                               f"Avg100={self.training_stats['average_reward']:.3f}, "
                               f"Epsilon={self.rl_selector.agent.epsilon:.3f}")
                
                # Save model periodically
                if (episode + 1) % self.config['save_interval'] == 0:
                    self.rl_selector.save_model()
                    logger.info(f"Model saved at episode {episode + 1}")
                
                # Evaluation
                if (episode + 1) % self.config['evaluation_interval'] == 0:
                    await self._evaluate_model(episode + 1)
                
                # Early stopping conditions
                if self._should_stop_training():
                    logger.info("Early stopping criteria met")
                    break
            
            # Final model save and evaluation
            self.rl_selector.save_model()
            await self._evaluate_model(self.training_stats['episodes'], final=True)
            
            # Generate training plots
            self._plot_training_progress()
            
            logger.info("RL training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    async def _run_episode(self, episode_num: int) -> float:
        """Run a single training episode"""
        total_episode_reward = 0.0
        episode_losses = []
        
        # Select a random learner for this episode
        learner_id = np.random.choice(self.learner_ids)
        
        for interaction in range(self.config['interactions_per_episode']):
            try:
                # Get next question from RL agent
                question_data = await self.rl_selector.select_next_question(learner_id)
                
                if not question_data or "error" in question_data:
                    continue
                
                interaction_id = question_data.get("interaction_id")
                if not interaction_id:
                    continue
                
                # Simulate learner answering the question
                await self._simulate_answer_submission(learner_id, question_data)
                
                # Get current state for feedback simulation
                current_state = await self.rl_selector.environment.get_current_state(learner_id)
                
                # Simulate user feedback
                feedback = await self.simulator.simulate_user_feedback(
                    learner_id=learner_id,
                    concept_id=question_data["concept_id"],
                    difficulty=question_data.get("difficulty", "medium"),
                    state=current_state
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
    
    async def _simulate_answer_submission(self, learner_id: str, question_data: Dict):
        """Simulate a learner submitting an answer"""
        concept_id = question_data["concept_id"]
        
        # Get current knowledge to simulate answer quality
        knowledge = await self.rl_selector.profile_manager.get_concept_knowledge(learner_id, concept_id)
        concept_score = (knowledge.get('current_score', 0.0) / 10.0) if knowledge else 0.0
        
        # Simulate answer accuracy based on knowledge and question difficulty
        difficulty_levels = {'easy': 0.3, 'medium': 0.6, 'hard': 0.9}
        question_difficulty = difficulty_levels.get(question_data.get("difficulty", "medium"), 0.6)
        
        # Probability of correct answer based on knowledge vs difficulty
        if concept_score >= question_difficulty:
            accuracy_prob = 0.8 + (concept_score - question_difficulty) * 0.2
        else:
            accuracy_prob = 0.3 + (concept_score / question_difficulty) * 0.4
        
        accuracy_prob = np.clip(accuracy_prob, 0.1, 0.95)
        simulated_accuracy = np.random.random() < accuracy_prob
        
        # Generate simulated score
        if simulated_accuracy:
            base_score = np.random.uniform(0.7, 1.0)
        else:
            base_score = np.random.uniform(0.1, 0.6)
        
        # Add some noise
        noise = np.random.normal(0, 0.1)
        final_score = np.clip(base_score + noise, 0.0, 1.0)
        
        # Update learner's knowledge (simulate the answer handler)
        graded_score = final_score * 10.0
        answered_correctly = final_score >= 0.7
        
        # Simple SRS update
        current_reps = knowledge.get('srs_repetitions', 0) if knowledge else 0
        current_interval = knowledge.get('srs_interval_days', 0) if knowledge else 0
        
        if answered_correctly:
            new_reps = current_reps + 1
            new_interval = max(1, current_interval * 2)
        else:
            new_reps = 0
            new_interval = 1
        
        srs_details = {
            'next_interval_days': new_interval,
            'next_review_at': datetime.now() + timedelta(days=new_interval),
            'new_srs_repetitions': new_reps
        }
        
        await self.rl_selector.profile_manager.update_concept_srs_and_difficulty(
            learner_id=learner_id,
            concept_id=concept_id,
            doc_id=question_data.get("doc_id", "training_doc"),
            score=graded_score,
            answered_correctly=answered_correctly,
            srs_details=srs_details
        )
    
    async def _evaluate_model(self, episode: int, final: bool = False):
        """Evaluate the current model performance"""
        logger.info(f"Evaluating model at episode {episode}...")
        
        # Temporarily disable training mode for evaluation
        original_epsilon = self.rl_selector.agent.epsilon
        self.rl_selector.agent.epsilon = 0.01  # Minimal exploration
        
        try:
            evaluation_rewards = []
            evaluation_feedback = {'up': 0, 'down': 0}
            
            # Run evaluation episodes
            for eval_episode in range(min(10, len(self.learner_ids))):
                learner_id = self.learner_ids[eval_episode]
                episode_reward = 0.0
                
                for _ in range(5):  # 5 questions per evaluation episode
                    try:
                        question_data = await self.rl_selector.select_next_question(learner_id)
                        
                        if not question_data or "error" in question_data:
                            continue
                        
                        interaction_id = question_data.get("interaction_id")
                        if not interaction_id:
                            continue
                        
                        # Simulate answer
                        await self._simulate_answer_submission(learner_id, question_data)
                        
                        # Get state for feedback
                        current_state = await self.rl_selector.environment.get_current_state(learner_id)
                        
                        # Simulate feedback
                        feedback = await self.simulator.simulate_user_feedback(
                            learner_id=learner_id,
                            concept_id=question_data["concept_id"],
                            difficulty=question_data.get("difficulty", "medium"),
                            state=current_state
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
            
            # Log evaluation results
            if evaluation_rewards:
                avg_eval_reward = np.mean(evaluation_rewards)
                total_feedback = evaluation_feedback['up'] + evaluation_feedback['down']
                upvote_rate = evaluation_feedback['up'] / total_feedback if total_feedback > 0 else 0
                
                logger.info(f"Evaluation Results (Episode {episode}):")
                logger.info(f"  Average Reward: {avg_eval_reward:.3f}")
                logger.info(f"  Upvote Rate: {upvote_rate:.2%}")
                logger.info(f"  Total Feedback: {total_feedback}")
                
                if final:
                    logger.info("=== FINAL EVALUATION ===")
                    logger.info(f"Total Episodes: {self.training_stats['episodes']}")
                    logger.info(f"Total Interactions: {self.training_stats['total_interactions']}")
                    logger.info(f"Final Average Reward: {avg_eval_reward:.3f}")
                    logger.info(f"Final Upvote Rate: {upvote_rate:.2%}")
                    logger.info(f"Training Time: {(time.time() - self.training_stats['training_start_time']) / 3600:.2f} hours")
        
        finally:
            # Restore original epsilon
            self.rl_selector.agent.epsilon = original_epsilon
    
    def _should_stop_training(self) -> bool:
        """Check if training should stop early"""
        # Check time limit
        if self.training_stats['training_start_time']:
            elapsed_hours = (time.time() - self.training_stats['training_start_time']) / 3600
            if elapsed_hours >= self.config['max_training_time_hours']:
                logger.info(f"Training time limit reached: {elapsed_hours:.2f} hours")
                return True
        
        # Check reward convergence
        if len(self.training_stats['episode_rewards']) >= 100:
            recent_rewards = self.training_stats['episode_rewards'][-100:]
            if np.mean(recent_rewards) >= self.config['target_average_reward']:
                logger.info(f"Target average reward reached: {np.mean(recent_rewards):.3f}")
                return True
        
        return False
    
    def _plot_training_progress(self):
        """Generate training progress plots"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Episode rewards
            if self.training_stats['episode_rewards']:
                episodes = range(1, len(self.training_stats['episode_rewards']) + 1)
                ax1.plot(episodes, self.training_stats['episode_rewards'], alpha=0.3, color='blue')
                
                # Moving average
                if len(self.training_stats['episode_rewards']) >= 10:
                    moving_avg = []
                    for i in range(9, len(self.training_stats['episode_rewards'])):
                        moving_avg.append(np.mean(self.training_stats['episode_rewards'][i-9:i+1]))
                    ax1.plot(range(10, len(self.training_stats['episode_rewards']) + 1), moving_avg, color='red', linewidth=2)
                
                ax1.set_title('Episode Rewards')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Total Reward')
                ax1.grid(True)
            
            # Training losses
            if self.training_stats['episode_losses']:
                episodes = range(1, len(self.training_stats['episode_losses']) + 1)
                ax2.plot(episodes, self.training_stats['episode_losses'], color='orange')
                ax2.set_title('Training Loss')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Loss')
                ax2.grid(True)
            
            # Feedback distribution
            total_up = self.training_stats['upvotes']
            total_down = self.training_stats['downvotes']
            if total_up + total_down > 0:
                ax3.pie([total_up, total_down], labels=['Upvotes', 'Downvotes'], 
                       colors=['green', 'red'], autopct='%1.1f%%')
                ax3.set_title('Feedback Distribution')
            
            # Agent statistics
            if hasattr(self.rl_selector, 'agent') and self.rl_selector.agent:
                agent_stats = self.rl_selector.agent.get_stats()
                stats_text = f"""
Training Steps: {agent_stats['training_steps']}
Episodes: {agent_stats['episodes']}
Epsilon: {agent_stats['epsilon']:.4f}
Buffer Size: {agent_stats['buffer_size']}
Recent Avg Loss: {agent_stats['recent_avg_loss']:.4f}
Device: {agent_stats['device']}
                """.strip()
                ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
                        verticalalignment='center', fontfamily='monospace')
                ax4.set_title('Agent Statistics')
                ax4.axis('off')
            
            plt.tight_layout()
            
            # Save plot
            os.makedirs(os.path.dirname(self.config['plots_save_path']), exist_ok=True)
            plt.savefig(self.config['plots_save_path'], dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training plots saved to {self.config['plots_save_path']}")
            
        except Exception as e:
            logger.error(f"Error generating training plots: {e}")
    
    def get_training_summary(self) -> Dict:
        """Get a summary of training results"""
        return {
            'episodes_completed': self.training_stats['episodes'],
            'total_interactions': self.training_stats['total_interactions'],
            'final_average_reward': self.training_stats['average_reward'],
            'final_average_loss': self.training_stats['average_loss'],
            'upvotes': self.training_stats['upvotes'],
            'downvotes': self.training_stats['downvotes'],
            'upvote_rate': self.training_stats['upvotes'] / max(1, self.training_stats['upvotes'] + self.training_stats['downvotes']),
            'training_time_hours': (time.time() - self.training_stats['training_start_time']) / 3600 if self.training_stats['training_start_time'] else 0,
            'agent_stats': self.rl_selector.agent.get_stats() if self.rl_selector and self.rl_selector.agent else {}
        }

async def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train RL Question Selector')
    parser.add_argument('--config', type=str, help='Path to training configuration file')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--learners', type=int, default=10, help='Number of virtual learners')
    parser.add_argument('--interactions', type=int, default=20, help='Interactions per episode')
    parser.add_argument('--save-interval', type=int, default=100, help='Model save interval')
    parser.add_argument('--eval-interval', type=int, default=50, help='Evaluation interval')
    parser.add_argument('--model-path', type=str, default='models/rl/question_selector.pt', help='Model save path')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = RLTrainer(args.config)
    
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
    
    # Ensure directories exist
    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('data/plots', exist_ok=True)
    os.makedirs(os.path.dirname(trainer.config['model_save_path']), exist_ok=True)
    
    try:
        # Initialize and run training
        await trainer.initialize()
        await trainer.run_training()
        
        # Print summary
        summary = trainer.get_training_summary()
        logger.info("=== TRAINING SUMMARY ===")
        for key, value in summary.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
        
        logger.info("RL training completed successfully!")
        
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

if __name__ == "__main__":
    # Setup environment
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run training
    asyncio.run(main())