# src/rl_engine/agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])

class DQN(nn.Module):
    """Deep Q-Network for question selection"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Network architecture
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward(self, state):
        """Forward pass through the network"""
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        return q_values

class ReplayBuffer:
    """Experience replay buffer for off-policy learning"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, experience: Experience):
        """Add an experience to the buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
    def can_sample(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size

class RLAgent:
    """
    Deep Q-Learning agent for adaptive question selection
    Implements Double DQN with experience replay
    """
    
    def __init__(self, 
                 state_size: int, 
                 action_size: int,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 100):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"RL Agent using device: {self.device}")
        
        # Neural networks
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training statistics
        self.training_steps = 0
        self.episodes = 0
        self.total_reward = 0.0
        self.losses = []
    
    def act(self, state: np.ndarray, valid_actions: Optional[List[int]] = None) -> int:
        """
        Choose an action using epsilon-greedy policy
        
        Args:
            state: Current state as numpy array
            valid_actions: List of valid action indices (optional)
        
        Returns:
            action_index: Selected action index
        """
        # Epsilon-greedy exploration
        if random.random() > self.epsilon:
            return self._exploit(state, valid_actions)
        else:
            return self._explore(valid_actions)
    
    def _exploit(self, state: np.ndarray, valid_actions: Optional[List[int]] = None) -> int:
        """Exploit: choose best action according to Q-network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        if valid_actions is not None:
            # Mask invalid actions with very negative values
            masked_q_values = q_values.clone()
            mask = torch.ones(self.action_size, dtype=torch.bool)
            mask[valid_actions] = False
            masked_q_values[0, mask] = -float('inf')
            action = masked_q_values.argmax().item()
        else:
            action = q_values.argmax().item()
        
        return action
    
    def _explore(self, valid_actions: Optional[List[int]] = None) -> int:
        """Explore: choose random action"""
        if valid_actions is not None:
            return random.choice(valid_actions)
        else:
            return random.randrange(self.action_size)
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.push(experience)
    
    def train(self) -> Optional[float]:
        """
        Train the agent using experience replay
        
        Returns:
            loss: Training loss if training occurred, None otherwise
        """
        if not self.replay_buffer.can_sample(self.batch_size):
            return None
        
        # Sample batch from replay buffer
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.info(f"Target network updated at step {self.training_steps}")
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def save_model(self, filepath: str):
        """Save the agent's model and training state"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episodes': self.episodes,
            'total_reward': self.total_reward,
            'losses': self.losses[-1000:],  # Keep last 1000 losses
            # Hyperparameters
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the agent's model and training state"""
        if not os.path.exists(filepath):
            logger.warning(f"Model file not found: {filepath}")
            return False
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.epsilon = checkpoint['epsilon']
            self.training_steps = checkpoint['training_steps']
            self.episodes = checkpoint['episodes']
            self.total_reward = checkpoint['total_reward']
            self.losses = checkpoint.get('losses', [])
            
            logger.info(f"Model loaded from {filepath}")
            logger.info(f"Training steps: {self.training_steps}, Episodes: {self.episodes}")
            logger.info(f"Current epsilon: {self.epsilon:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get training statistics"""
        return {
            'training_steps': self.training_steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'total_reward': self.total_reward,
            'buffer_size': len(self.replay_buffer),
            'recent_avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'device': str(self.device)
        }