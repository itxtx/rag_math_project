# src/rl_engine/__init__.py
"""
RL Engine Package for Adaptive Question Selection

This package implements a Hybrid Reinforcement Learning system that replaces
the rule-based QuestionSelector with a dynamic, engagement-driven RL agent.

Components:
- RLEnvironment: Manages state representation and action space
- RLAgent: Deep Q-Network implementation with experience replay
- RewardSystem: Hybrid reward function with engagement, learning, and effort components
- RLQuestionSelector: Main interface replacing the original QuestionSelector
- Training utilities: Scripts and simulators for offline training
"""

from .environment import RLEnvironment, State, Action, DifficultyLevel
from .agent import RLAgent, ReplayBuffer
from .reward_system import (
    HybridRewardCalculator, 
    RewardSystemManager, 
    VoteType, 
    RewardComponents,
    InteractionTracker
)
from .rl_question_selector import RLQuestionSelector

__all__ = [
    'RLEnvironment',
    'State', 
    'Action',
    'DifficultyLevel',
    'RLAgent',
    'ReplayBuffer',
    'HybridRewardCalculator',
    'RewardSystemManager',
    'VoteType',
    'RewardComponents', 
    'InteractionTracker',
    'RLQuestionSelector'
]

# Configuration constants
DEFAULT_RL_CONFIG = {
    'model_path': 'models/rl/question_selector.pt',
    'state_features': {
        'global_features': 4,  # last_score, session_length, avg_engagement, time_since_last
        'per_concept_features': 3,  # score, attempts, is_last
        'difficulty_features': 3,  # one-hot encoding
    },
    'reward_weights': {
        'vote': 0.6,
        'learning': 0.3, 
        'effort': 0.1
    },
    'agent_hyperparameters': {
        'learning_rate': 1e-4,
        'gamma': 0.95,
        'epsilon_start': 0.1,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 50000,
        'batch_size': 64,
        'target_update_freq': 1000,
        'hidden_size': 256
    },
    'training_config': {
        'episodes': 1000,
        'interactions_per_episode': 20,
        'virtual_learners': 10,
        'save_interval': 100,
        'eval_interval': 50,
        'max_training_hours': 24
    }
}

# src/rl_engine/config.py
import os
import json
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class RLConfig:
    """Configuration class for RL system"""
    
    # Model paths
    model_save_path: str = "models/rl/question_selector.pt"
    config_save_path: str = "models/rl/config.json"
    
    # Reward system weights
    reward_weight_vote: float = 0.6
    reward_weight_learning: float = 0.3
    reward_weight_effort: float = 0.1
    
    # Agent hyperparameters
    learning_rate: float = 1e-4
    gamma: float = 0.95
    epsilon_start: float = 0.1
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 50000
    batch_size: int = 64
    target_update_freq: int = 1000
    hidden_size: int = 256
    
    # Training parameters
    training_episodes: int = 1000
    interactions_per_episode: int = 20
    num_virtual_learners: int = 10
    save_interval: int = 100
    evaluation_interval: int = 50
    max_training_time_hours: int = 24
    
    # Environment settings
    interaction_timeout_minutes: int = 5
    max_session_length: int = 50
    concept_mastery_threshold: float = 0.9
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RLConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    def save(self, filepath: str = None):
        """Save configuration to JSON file"""
        save_path = filepath or self.config_save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'RLConfig':
        """Load configuration from JSON file"""
        if not os.path.exists(filepath):
            return cls()  # Return default config
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)

# src/rl_engine/utils.py
import numpy as np
import torch
import logging
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RLMetrics:
    """Utility class for RL metrics and visualization"""
    
    @staticmethod
    def calculate_moving_average(values: List[float], window: int = 100) -> List[float]:
        """Calculate moving average of values"""
        if len(values) < window:
            return values
        
        moving_avg = []
        for i in range(window - 1, len(values)):
            avg = np.mean(values[i - window + 1:i + 1])
            moving_avg.append(avg)
        
        return moving_avg
    
    @staticmethod
    def plot_training_metrics(rewards: List[float], 
                            losses: List[float],
                            upvotes: int,
                            downvotes: int,
                            save_path: str = None) -> None:
        """Plot comprehensive training metrics"""
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create a 3x2 subplot layout
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Episode rewards with moving average
        ax1 = fig.add_subplot(gs[0, 0])
        if rewards:
            episodes = range(1, len(rewards) + 1)
            ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Rewards')
            
            if len(rewards) >= 20:
                moving_avg = RLMetrics.calculate_moving_average(rewards, 20)
                ax1.plot(range(20, len(rewards) + 1), moving_avg, 
                        color='red', linewidth=2, label='Moving Average (20)')
            
            ax1.set_title('Episode Rewards Over Time')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Training losses
        ax2 = fig.add_subplot(gs[0, 1])
        if losses:
            episodes = range(1, len(losses) + 1)
            ax2.plot(episodes, losses, color='orange', alpha=0.7)
            
            if len(losses) >= 10:
                moving_avg = RLMetrics.calculate_moving_average(losses, 10)
                ax2.plot(range(10, len(losses) + 1), moving_avg, 
                        color='red', linewidth=2, label='Moving Average (10)')
            
            ax2.set_title('Training Loss')
            ax2.set_xlabel('Episode') 
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Feedback distribution pie chart
        ax3 = fig.add_subplot(gs[1, 0])
        if upvotes + downvotes > 0:
            labels = ['Upvotes', 'Downvotes']
            sizes = [upvotes, downvotes]
            colors = ['#2ecc71', '#e74c3c']
            
            wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, 
                                             autopct='%1.1f%%', startangle=90)
            ax3.set_title('User Feedback Distribution')
            
            # Add count annotations
            total_feedback = upvotes + downvotes
            ax3.text(0, -1.3, f'Total Feedback: {total_feedback}', 
                    ha='center', transform=ax3.transAxes, fontsize=10)
        
        # 4. Reward distribution histogram
        ax4 = fig.add_subplot(gs[1, 1])
        if rewards:
            ax4.hist(rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.axvline(np.mean(rewards), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(rewards):.3f}')
            ax4.axvline(np.median(rewards), color='green', linestyle='--',
                       label=f'Median: {np.median(rewards):.3f}')
            ax4.set_title('Reward Distribution')
            ax4.set_xlabel('Reward')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Learning progress (reward trend)
        ax5 = fig.add_subplot(gs[2, :])
        if rewards:
            # Split into training phases
            phases = ['Early', 'Mid', 'Late']
            phase_size = len(rewards) // 3
            
            early_rewards = rewards[:phase_size] if phase_size > 0 else []
            mid_rewards = rewards[phase_size:2*phase_size] if phase_size > 0 else []
            late_rewards = rewards[2*phase_size:] if phase_size > 0 else rewards
            
            phase_data = []
            phase_labels = []
            
            if early_rewards:
                phase_data.append(early_rewards)
                phase_labels.append(f'Early (1-{len(early_rewards)})')
            if mid_rewards:
                phase_data.append(mid_rewards)
                phase_labels.append(f'Mid ({len(early_rewards)+1}-{len(early_rewards)+len(mid_rewards)})')
            if late_rewards:
                phase_data.append(late_rewards)
                phase_labels.append(f'Late ({len(early_rewards)+len(mid_rewards)+1}-{len(rewards)})')
            
            if phase_data:
                box_plot = ax5.boxplot(phase_data, labels=phase_labels, patch_artist=True)
                colors = ['lightblue', 'lightgreen', 'lightcoral']
                for patch, color in zip(box_plot['boxes'], colors[:len(phase_data)]):
                    patch.set_facecolor(color)
                
                ax5.set_title('Learning Progress Across Training Phases')
                ax5.set_ylabel('Reward')
                ax5.grid(True, alpha=0.3)
        
        # Save plot
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training metrics plot saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def evaluate_policy_performance(agent, environment, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the performance of a trained policy"""
        total_rewards = []
        q_value_stats = []
        action_entropy = []
        
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0  # No exploration during evaluation
        
        try:
            for episode in range(num_episodes):
                episode_reward = 0.0
                episode_q_values = []
                episode_actions = []
                
                # Simulate episode
                for step in range(20):  # 20 steps per episode
                    # Generate random state for evaluation
                    state = np.random.random(environment.state_size)
                    
                    # Get Q-values
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        q_values = agent.q_network(state_tensor)
                        q_values_np = q_values.cpu().numpy()[0]
                    
                    # Select action
                    action = agent.act(state)
                    
                    episode_q_values.extend(q_values_np)
                    episode_actions.append(action)
                    
                    # Simulate reward (simplified)
                    reward = np.random.normal(0.5, 0.2)
                    episode_reward += reward
                
                total_rewards.append(episode_reward)
                q_value_stats.extend(episode_q_values)
                
                # Calculate action entropy for this episode
                action_counts = np.bincount(episode_actions, minlength=environment.action_space_size)
                action_probs = action_counts / len(episode_actions)
                entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
                action_entropy.append(entropy)
        
        finally:
            agent.epsilon = original_epsilon
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_q_value': np.mean(q_value_stats),
            'std_q_value': np.std(q_value_stats),
            'mean_action_entropy': np.mean(action_entropy),
            'reward_stability': np.std(total_rewards) / (np.mean(total_rewards) + 1e-8)
        }

class RLMonitor:
    """Real-time monitoring for RL system"""
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.metrics = {
            'interactions': 0,
            'rewards': [],
            'losses': [],
            'upvotes': 0,
            'downvotes': 0,
            'start_time': datetime.now(),
            'last_log_time': datetime.now()
        }
    
    def log_interaction(self, reward: float, feedback_type: str, loss: float = None):
        """Log a single interaction"""
        self.metrics['interactions'] += 1
        self.metrics['rewards'].append(reward)
        
        if loss is not None:
            self.metrics['losses'].append(loss)
        
        if feedback_type == 'up':
            self.metrics['upvotes'] += 1
        elif feedback_type == 'down':
            self.metrics['downvotes'] += 1
        
        # Log periodic updates
        if self.metrics['interactions'] % self.log_interval == 0:
            self._log_periodic_update()
    
    def _log_periodic_update(self):
        """Log periodic performance update"""
        current_time = datetime.now()
        time_elapsed = current_time - self.metrics['start_time']
        
        recent_rewards = self.metrics['rewards'][-self.log_interval:]
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        
        total_feedback = self.metrics['upvotes'] + self.metrics['downvotes']
        upvote_rate = self.metrics['upvotes'] / total_feedback if total_feedback > 0 else 0.0
        
        interactions_per_hour = self.metrics['interactions'] / (time_elapsed.total_seconds() / 3600)
        
        logger.info(f"RL Monitor Update - Interactions: {self.metrics['interactions']}")
        logger.info(f"  Recent Avg Reward: {avg_reward:.3f}")
        logger.info(f"  Upvote Rate: {upvote_rate:.2%}")
        logger.info(f"  Interactions/Hour: {interactions_per_hour:.1f}")
        logger.info(f"  Total Time: {time_elapsed}")
        
        self.metrics['last_log_time'] = current_time
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete monitoring summary"""
        current_time = datetime.now()
        total_time = current_time - self.metrics['start_time']
        
        total_feedback = self.metrics['upvotes'] + self.metrics['downvotes']
        
        return {
            'total_interactions': self.metrics['interactions'],
            'total_time_hours': total_time.total_seconds() / 3600,
            'interactions_per_hour': self.metrics['interactions'] / (total_time.total_seconds() / 3600),
            'average_reward': np.mean(self.metrics['rewards']) if self.metrics['rewards'] else 0.0,
            'reward_std': np.std(self.metrics['rewards']) if self.metrics['rewards'] else 0.0,
            'upvotes': self.metrics['upvotes'],
            'downvotes': self.metrics['downvotes'],
            'upvote_rate': self.metrics['upvotes'] / total_feedback if total_feedback > 0 else 0.0,
            'average_loss': np.mean(self.metrics['losses']) if self.metrics['losses'] else 0.0,
            'recent_reward_trend': np.mean(self.metrics['rewards'][-100:]) if len(self.metrics['rewards']) >= 100 else None
        }

# src/rl_engine/deployment.py
import asyncio
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class RLSystemManager:
    """
    High-level manager for deploying and managing the RL system in production
    """
    
    def __init__(self, 
                 profile_manager,
                 retriever, 
                 question_generator,
                 config: Optional[RLConfig] = None):
        
        self.config = config or RLConfig()
        self.rl_selector = RLQuestionSelector(
            profile_manager=profile_manager,
            retriever=retriever,
            question_generator=question_generator,
            model_path=self.config.model_save_path
        )
        
        self.monitor = RLMonitor()
        self.is_running = False
        self.deployment_time = None
        
    async def deploy(self, training_mode: bool = False):
        """Deploy the RL system"""
        try:
            logger.info("Deploying RL Question Selection System...")
            
            # Initialize the RL selector
            await self.rl_selector.initialize()
            
            # Set training mode
            if training_mode:
                self.rl_selector.enable_training_mode()
                logger.info("RL system deployed in TRAINING mode")
            else:
                self.rl_selector.disable_training_mode()
                logger.info("RL system deployed in PRODUCTION mode")
            
            self.is_running = True
            self.deployment_time = datetime.now()
            
            # Log deployment stats
            stats = self.rl_selector.get_stats()
            logger.info(f"RL System deployed successfully:")
            logger.info(f"  Available concepts: {stats.get('available_concepts', 0)}")
            logger.info(f"  Model initialized: {stats.get('initialized', False)}")
            logger.info(f"  Training mode: {stats.get('training_mode', False)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy RL system: {e}")
            self.is_running = False
            return False
    
    async def select_question(self, learner_id: str, target_doc_id: Optional[str] = None):
        """Select next question using RL system"""
        if not self.is_running:
            raise RuntimeError("RL system not running. Call deploy() first.")
        
        try:
            question_data = await self.rl_selector.select_next_question(learner_id, target_doc_id)
            return question_data
            
        except Exception as e:
            logger.error(f"Error in RL question selection: {e}")
            return {"error": f"RL selection failed: {str(e)}"}
    
    async def process_feedback(self, 
                             interaction_id: str,
                             learner_id: str, 
                             concept_id: str,
                             feedback_type: str) -> bool:
        """Process user feedback"""
        if not self.is_running:
            return False
        
        try:
            vote_type = VoteType.UPVOTE if feedback_type == 'up' else VoteType.DOWNVOTE
            
            success = await self.rl_selector.process_feedback(
                interaction_id=interaction_id,
                learner_id=learner_id,
                concept_id=concept_id,
                vote_type=vote_type
            )
            
            if success:
                # Get reward for monitoring
                interaction_data = self.rl_selector.reward_manager.interaction_tracker.get_interaction_data(interaction_id)
                reward = 0.0
                loss = None
                
                if interaction_data and interaction_data.get('completed'):
                    reward = interaction_data['reward_components'].total_reward
                
                # Get recent loss if available
                if self.rl_selector.agent and self.rl_selector.agent.losses:
                    loss = self.rl_selector.agent.losses[-1]
                
                # Log to monitor
                self.monitor.log_interaction(reward, feedback_type, loss)
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing RL feedback: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        base_status = {
            'is_running': self.is_running,
            'deployment_time': self.deployment_time.isoformat() if self.deployment_time else None,
            'config': self.config.to_dict()
        }
        
        if self.is_running and self.rl_selector:
            rl_stats = self.rl_selector.get_stats()
            monitor_stats = self.monitor.get_summary()
            
            base_status.update({
                'rl_system': rl_stats,
                'monitoring': monitor_stats
            })
        
        return base_status
    
    async def save_model(self):
        """Save the current RL model"""
        if self.is_running and self.rl_selector:
            self.rl_selector.save_model()
            self.config.save()
            logger.info("RL model and config saved")
    
    async def shutdown(self):
        """Gracefully shutdown the RL system"""
        logger.info("Shutting down RL system...")
        
        if self.is_running and self.rl_selector:
            # Save model before shutdown
            await self.save_model()
            
            # Log final statistics
            final_stats = self.monitor.get_summary()
            logger.info("Final RL System Statistics:")
            for key, value in final_stats.items():
                logger.info(f"  {key}: {value}")
        
        self.is_running = False
        logger.info("RL system shutdown complete")

# Example integration script: scripts/deploy_rl_system.py
"""
#!/usr/bin/env python3
# scripts/deploy_rl_system.py

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rl_engine.deployment import RLSystemManager, RLConfig
from src.data_ingestion import vector_store_manager
from src.learner_model.profile_manager import LearnerProfileManager
from src.retrieval.retriever import HybridRetriever
from src.generation.question_generator_rag import RAGQuestionGenerator

async def main():
    # Initialize components
    weaviate_client = vector_store_manager.get_weaviate_client()
    profile_manager = LearnerProfileManager()
    retriever = HybridRetriever(weaviate_client=weaviate_client)
    question_generator = RAGQuestionGenerator()
    
    # Load config
    config = RLConfig.load('models/rl/config.json')
    
    # Create RL system manager
    rl_manager = RLSystemManager(
        profile_manager=profile_manager,
        retriever=retriever,
        question_generator=question_generator,
        config=config
    )
    
    # Deploy system
    success = await rl_manager.deploy(training_mode=False)
    
    if success:
        print("RL system deployed successfully!")
        
        # Example usage
        question_data = await rl_manager.select_question("test_learner_001")
        print(f"Selected question: {question_data}")
        
        # Get system status
        status = rl_manager.get_system_status()
        print(f"System status: {status}")
        
        # Shutdown
        await rl_manager.shutdown()
    else:
        print("Failed to deploy RL system")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
"""

# Update to Makefile - Add RL training commands
"""
# Add these commands to the existing Makefile

# RL System Commands
rl-train:
	@echo "$(BLUE)🤖 Training RL Question Selector...$(NC)"
	@echo "$(YELLOW)This trains the AI to learn optimal teaching strategies$(NC)"
	@python -m src.rl_engine.train --episodes 1000 --learners 10

rl-train-quick:
	@echo "$(BLUE)🤖 Quick RL training (100 episodes)...$(NC)"
	@python -m src.rl_engine.train --episodes 100 --learners 5 --interactions 10

rl-deploy:
	@echo "$(BLUE)🚀 Deploying RL system...$(NC)"
	@python scripts/deploy_rl_system.py

rl-stats:
	@echo "$(BLUE)📊 RL System Statistics$(NC)"
	@curl -s http://localhost:8000/api/v1/rl/stats | python -m json.tool

rl-enable-training:
	@echo "$(BLUE)🎓 Enabling RL training mode...$(NC)"
	@curl -X POST http://localhost:8000/api/v1/rl/training/enable

rl-disable-training:
	@echo "$(BLUE)🎯 Disabling RL training mode (production)...$(NC)"
	@curl -X POST http://localhost:8000/api/v1/rl/training/disable

rl-save:
	@echo "$(BLUE)💾 Saving RL model...$(NC)"
	@curl -X POST http://localhost:8000/api/v1/rl/model/save

# Complete RL workflow
rl-setup: setup rl-train rl-deploy
	@echo "$(GREEN)🎉 Complete RL system setup finished!$(NC)"
"""