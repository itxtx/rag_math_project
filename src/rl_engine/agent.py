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

class DeviceManager:
    """Manages consistent device placement for tensors"""
    
    def __init__(self, device: torch.device):
        self.device = device
        logger.info(f"DeviceManager initialized with device: {device}")
    
    def to_device(self, tensor_or_array) -> torch.Tensor:
        """
        Ensure tensor is on the correct device
        
        Args:
            tensor_or_array: Input tensor or numpy array
            
        Returns:
            torch.Tensor: Tensor on the correct device
        """
        if isinstance(tensor_or_array, torch.Tensor):
            # If it's already a tensor, move it to the correct device
            if tensor_or_array.device != self.device:
                return tensor_or_array.to(self.device)
            return tensor_or_array
        elif isinstance(tensor_or_array, np.ndarray):
            # If it's a numpy array, convert to tensor and move to device
            return torch.FloatTensor(tensor_or_array).to(self.device)
        else:
            # For other types, try to convert to tensor
            try:
                return torch.FloatTensor(tensor_or_array).to(self.device)
            except Exception as e:
                logger.error(f"Failed to convert {type(tensor_or_array)} to tensor: {e}")
                raise
    
    def batch_to_device(self, batch_data) -> torch.Tensor:
        """
        Convert a batch of data to tensors on the correct device
        
        Args:
            batch_data: List or array of data
            
        Returns:
            torch.Tensor: Batch tensor on the correct device
        """
        if isinstance(batch_data, list):
            # Convert list of arrays/tensors to batch tensor
            tensors = [self.to_device(item) for item in batch_data]
            return torch.stack(tensors) if tensors else torch.empty(0).to(self.device)
        else:
            # Single item
            return self.to_device(batch_data)
    
    def get_device_info(self) -> dict:
        """Get information about the current device"""
        return {
            'device': str(self.device),
            'device_type': self.device.type,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

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

class RLAgentError(Exception):
    """Base exception for RL agent errors"""
    pass

class DeviceError(RLAgentError):
    """Exception for device-related errors"""
    pass

class TrainingError(RLAgentError):
    """Exception for training-related errors"""
    pass

class ValidationError(RLAgentError):
    """Exception for validation errors"""
    pass

class MemoryError(RLAgentError):
    """Exception for memory-related errors"""
    pass

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
        
        # FIXED: Enhanced device selection with automatic detection
        self.device = self._select_best_device()
        logger.info(f"RL Agent using device: {self.device}")
        
        # FIXED: Add global device management
        self._device_manager = DeviceManager(self.device)
        
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
        def _exploit_operation():
            # FIXED: Use device manager to ensure consistent device placement
            state_tensor = self._device_manager.to_device(state).unsqueeze(0)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            
            if valid_actions is not None:
                # Mask invalid actions with very negative values
                masked_q_values = q_values.clone()
                mask = torch.ones(self.action_size, dtype=torch.bool, device=self.device)
                mask[valid_actions] = False
                masked_q_values[0, mask] = -float('inf')
                action = masked_q_values.argmax().item()
            else:
                action = q_values.argmax().item()
            
            return action
        
        # FIXED: Use safe operation wrapper for device error handling
        try:
            return self.safe_tensor_operation(_exploit_operation)
        except DeviceError as e:
            logger.error(f"Device error in exploit: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in exploit: {e}")
            raise TrainingError(f"Exploit operation failed: {e}") from e
    
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
        
        # FIXED: Use device manager for consistent device placement
        states = self._device_manager.batch_to_device([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = self._device_manager.batch_to_device([e.next_state for e in experiences])
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
    
    def validate_device_consistency(self) -> bool:
        """Validate that all components are on the correct device"""
        try:
            # Check networks
            q_network_device = next(self.q_network.parameters()).device
            target_network_device = next(self.target_network.parameters()).device
            
            if q_network_device != self.device:
                logger.error(f"Q-network on wrong device: {q_network_device}, expected: {self.device}")
                return False
            
            if target_network_device != self.device:
                logger.error(f"Target network on wrong device: {target_network_device}, expected: {self.device}")
                return False
            
            # Check optimizer
            optimizer_device = next(self.optimizer.param_groups[0]['params']).device
            if optimizer_device != self.device:
                logger.error(f"Optimizer on wrong device: {optimizer_device}, expected: {self.device}")
                return False
            
            logger.debug("Device consistency validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error during device consistency validation: {e}")
            return False
    
    def ensure_device_consistency(self) -> bool:
        """Ensure all components are on the correct device, migrate if needed"""
        if self.validate_device_consistency():
            return True
        
        logger.warning("Device inconsistency detected, attempting to fix...")
        return self.migrate_to_device(self.device)
    
    def migrate_to_device(self, new_device: torch.device) -> bool:
        """Migrate all components to a new device"""
        try:
            logger.info(f"Migrating RL agent from {self.device} to {new_device}")
            
            # Update device manager
            self.device = new_device
            self._device_manager = DeviceManager(new_device)
            
            # Move networks to new device
            self.q_network = self.q_network.to(new_device)
            self.target_network = self.target_network.to(new_device)
            
            # Move optimizer parameters to new device
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    param.data = param.data.to(new_device)
            
            # Validate migration
            if self.validate_device_consistency():
                logger.info(f"Successfully migrated to {new_device}")
                return True
            else:
                logger.error(f"Device migration validation failed")
                return False
                
        except Exception as e:
            logger.error(f"Error during device migration: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get training statistics"""
        # FIXED: Add device information and validation
        device_consistent = self.validate_device_consistency()
        
        return {
            'training_steps': self.training_steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'total_reward': self.total_reward,
            'buffer_size': len(self.replay_buffer),
            'recent_avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'device': str(self.device),
            'device_consistent': device_consistent,
            'device_info': self._device_manager.get_device_info()
        }

    def _select_best_device(self):
        """Select the best available device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def handle_device_error(self, error: Exception) -> bool:
        """Handle device-related errors gracefully"""
        error_str = str(error).lower()
        
        if "cuda" in error_str or "gpu" in error_str:
            logger.warning("CUDA error detected, falling back to CPU")
            return self.migrate_to_device(torch.device("cpu"))
        elif "mps" in error_str:
            logger.warning("MPS error detected, falling back to CPU")
            return self.migrate_to_device(torch.device("cpu"))
        elif "out of memory" in error_str:
            logger.warning("Out of memory error, trying to free cache and continue")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True  # Try to continue
        else:
            logger.error(f"Unknown device error: {error}")
            return False
    
    def safe_tensor_operation(self, operation_func, *args, **kwargs):
        """Safely execute tensor operations with device error handling"""
        # FIXED: Ensure device consistency before operations
        if not self.ensure_device_consistency():
            logger.error("Failed to ensure device consistency")
            raise RuntimeError("Device consistency check failed")
        
        try:
            return operation_func(*args, **kwargs)
        except Exception as e:
            if self.handle_device_error(e):
                # Retry the operation after device migration
                try:
                    return operation_func(*args, **kwargs)
                except Exception as retry_error:
                    logger.error(f"Operation failed even after device migration: {retry_error}")
                    raise
            else:
                raise

    def _calculate_confidence(self, state_vector: np.ndarray, action_index: int) -> float:
        """Calculate confidence in the selected action"""
        try:
            if not self.agent:
                raise ValidationError("Agent not initialized")
            
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.agent.device)
            
            with torch.no_grad():
                q_values = self.agent.q_network(state_tensor)
                q_values_np = q_values.cpu().numpy()[0]
                
                # FIXED: Improved confidence calculation
                if len(q_values_np) <= action_index:
                    raise ValidationError(f"Action index {action_index} out of bounds for Q-values of length {len(q_values_np)}")
                
                # Confidence based on Q-value difference and distribution
                max_q = np.max(q_values_np)
                selected_q = q_values_np[action_index]
                min_q = np.min(q_values_np)
                
                # Normalize Q-values to 0-1 range
                q_range = max_q - min_q
                if q_range == 0:
                    return 0.5  # Neutral confidence if all Q-values are equal
                
                normalized_selected = (selected_q - min_q) / q_range
                
                # Additional confidence from how much better this action is than the average
                mean_q = np.mean(q_values_np)
                relative_advantage = (selected_q - mean_q) / (q_range + 1e-8)
                
                # Combine normalized value and relative advantage
                confidence = 0.7 * normalized_selected + 0.3 * (0.5 + relative_advantage)
                confidence = max(0.1, min(0.95, confidence))  # Bound between 0.1 and 0.95
                
                return confidence
                
        except ValidationError:
            raise
        except DeviceError as e:
            logger.error(f"Device error calculating confidence: {e}")
            raise
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            raise TrainingError(f"Confidence calculation failed: {e}") from e