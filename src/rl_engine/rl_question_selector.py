# src/rl_engine/rl_question_selector.py
import asyncio
import uuid
import logging
from typing import Optional, Dict, Any, List
import numpy as np
import os
import torch
from .environment import RLEnvironment, State, Action, DifficultyLevel
from .agent import RLAgent
from .reward_system import RewardSystemManager, VoteType
from src.generation.question_generator_rag import RAGQuestionGenerator

logger = logging.getLogger(__name__)

class RLQuestionSelector:
    """
    RL-powered question selector that replaces the rule-based QuestionSelector
    Uses Deep Q-Learning to learn optimal teaching policies
    """
    
    def __init__(self, 
                 profile_manager,
                 retriever,
                 question_generator: RAGQuestionGenerator,
                 model_path: str = "models/rl/question_selector.pt"):
        
        self.profile_manager = profile_manager
        self.retriever = retriever
        self.question_generator = question_generator
        self.model_path = model_path
        
        # Initialize RL components
        self.environment = RLEnvironment(profile_manager, retriever)
        self.agent = None  # Will be initialized after environment
        self.reward_manager = RewardSystemManager(profile_manager)
        
        self.is_initialized = False
        self.training_mode = False
        
        logger.info("RLQuestionSelector created, awaiting initialization")
    
    async def initialize(self):
        """Initialize the RL components"""
        try:
            logger.info("Initializing RL Question Selector...")
            
            # Initialize environment first
            await self.environment.initialize()
            
            # Create RL agent with proper dimensions
            self.agent = RLAgent(
                state_size=self.environment.state_size,
                action_size=self.environment.action_space_size,
                learning_rate=1e-4,  # Lower learning rate for stable learning
                gamma=0.95,
                epsilon_start=0.1,   # Start with low exploration for production
                epsilon_end=0.01,
                epsilon_decay=0.995,
                buffer_size=50000,   # Larger buffer for better experience replay
                batch_size=64,
                target_update_freq=1000
            )
            
            # Try to load existing model
            if os.path.exists(self.model_path):
                if self.agent.load_model(self.model_path):
                    logger.info("Loaded existing RL model")
                else:
                    logger.warning("Failed to load existing model, starting fresh")
            else:
                logger.info("No existing model found, starting with random weights")
            
            self.is_initialized = True
            logger.info("RL Question Selector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RL Question Selector: {e}")
            raise
    
    async def select_next_question(self, 
                                 learner_id: str, 
                                 target_doc_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Select the next question using the RL agent
        
        Args:
            learner_id: ID of the learner
            target_doc_id: Optional document ID to restrict selection
            
        Returns:
            Question data dictionary or None if no question can be selected
        """
        if not self.is_initialized:
            logger.error("RL Question Selector not initialized")
            return {"error": "RL system not initialized"}
        
        try:
            logger.info(f"RL agent selecting question for learner {learner_id}")
            
            # Get current state
            current_state = await self.environment.get_current_state(learner_id)
            state_vector = current_state.to_feature_vector(self.environment.available_concepts)
            
            # Get valid actions (filter by target_doc_id if specified)
            valid_actions = self._get_valid_actions(current_state, target_doc_id)
            
            if not valid_actions:
                logger.warning(f"No valid actions available for learner {learner_id}")
                return {"error": "No suitable concepts available"}
            
            # Agent selects action
            action_index = self.agent.act(state_vector, valid_actions)
            selected_action = self.environment.action_index_to_action(action_index)
            
            logger.info(f"RL agent selected: concept={selected_action.concept_id}, "
                       f"difficulty={selected_action.difficulty_level.value}")
            
            # Generate question for the selected concept
            question_data = await self._generate_question_for_action(
                selected_action, learner_id, current_state
            )
            
            if not question_data:
                return {"error": "Failed to generate question for selected concept"}
            
            # Start tracking this interaction for reward calculation
            interaction_id = str(uuid.uuid4())
            await self.reward_manager.start_interaction(
                interaction_id=interaction_id,
                learner_id=learner_id,
                concept_id=selected_action.concept_id,
                difficulty=selected_action.difficulty_level.value,
                state_before=state_vector
            )
            
            # Add RL-specific metadata to question response
            question_data.update({
                "interaction_id": interaction_id,
                "selected_by_rl": True,
                "rl_action": selected_action.to_dict(),
                "rl_confidence": self._calculate_confidence(state_vector, action_index)
            })
            
            return question_data
            
        except Exception as e:
            logger.error(f"Error in RL question selection: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"RL selection failed: {str(e)}"}
    
    def _get_valid_actions(self, state: State, target_doc_id: Optional[str] = None) -> List[int]:
        """Get valid actions, optionally filtered by document ID"""
        valid_actions = self.environment.get_valid_actions(state)
        
        if target_doc_id is None:
            return valid_actions
        
        # Filter by target document
        filtered_actions = []
        for action_idx in valid_actions:
            action = self.environment.action_index_to_action(action_idx)
            # Check if concept belongs to target document
            # This requires mapping concepts to documents - simplified for now
            filtered_actions.append(action_idx)
        
        return filtered_actions if filtered_actions else valid_actions
    
    async def _generate_question_for_action(self, 
                                          action: Action, 
                                          learner_id: str,
                                          state: State) -> Optional[Dict[str, Any]]:
        """Generate a question for the selected action"""
        try:
            # Get context chunks for the concept
            context_chunks = await self.retriever.get_chunks_for_parent_block(
                action.concept_id, limit=3
            )
            
            if not context_chunks:
                logger.warning(f"No context found for concept {action.concept_id}")
                return None
            
            # Determine question style based on learner progress
            concept_score = state.concept_scores.get(action.concept_id, 0.0)
            attempts = state.concept_attempts.get(action.concept_id, 0)
            
            # Adaptive question styling
            if attempts == 0:
                question_type = "definition_recall"
                question_style = "standard"
            elif concept_score < 0.3:
                question_type = "conceptual"
                question_style = "fill_in_blank"
            elif concept_score < 0.7:
                question_type = "application"
                question_style = "standard"
            else:
                question_type = "reasoning"
                question_style = "complete_proof_step"
            
            # Generate question
            questions = await self.question_generator.generate_questions(
                context_chunks=context_chunks,
                num_questions=1,
                question_type=question_type,
                difficulty_level=action.difficulty_level.value,
                question_style=question_style
            )
            
            if not questions:
                logger.error(f"Question generation failed for concept {action.concept_id}")
                return None
            
            # Get concept name
            concept_name = context_chunks[0].get('concept_name', 'Unknown Concept')
            doc_id = context_chunks[0].get('doc_id', 'unknown_doc')
            
            # Build context for evaluation
            context_for_evaluation = "\n\n".join([
                chunk.get('chunk_text', '') for chunk in context_chunks
            ])
            
            return {
                "learner_id": learner_id,
                "doc_id": doc_id,
                "concept_id": action.concept_id,
                "concept_name": concept_name,
                "question_text": questions[0],
                "context_chunks": [chunk['chunk_text'] for chunk in context_chunks],
                "context_for_evaluation": context_for_evaluation,
                "is_review": attempts > 0,
                "difficulty": action.difficulty_level.value,
                "question_type": question_type,
                "question_style": question_style,
                "is_new_concept_context_presented": attempts == 0
            }
            
        except Exception as e:
            logger.error(f"Error generating question for action: {e}")
            return None
    
    def _calculate_confidence(self, state_vector: np.ndarray, action_index: int) -> float:
        """Calculate confidence in the selected action"""
        try:
            if not self.agent:
                return 0.5
            
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.agent.device)
            
            with torch.no_grad():
                q_values = self.agent.q_network(state_tensor)
                q_values_np = q_values.cpu().numpy()[0]
                
                # Confidence based on Q-value difference
                max_q = np.max(q_values_np)
                selected_q = q_values_np[action_index]
                
                if max_q == selected_q:
                    # Selected action has highest Q-value
                    second_max = np.partition(q_values_np, -2)[-2]
                    confidence = min(1.0, (max_q - second_max + 1.0) / 2.0)
                else:
                    # Selected action is suboptimal (exploration)
                    confidence = min(1.0, (selected_q - np.min(q_values_np) + 1.0) / 2.0)
                
                return max(0.1, confidence)  # Minimum confidence of 0.1
                
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    async def process_feedback(self, 
                             interaction_id: str,
                             learner_id: str,
                             concept_id: str,
                             vote_type: VoteType) -> bool:
        """
        Process user feedback and trigger reward calculation
        
        Args:
            interaction_id: ID of the interaction
            learner_id: ID of the learner
            concept_id: ID of the concept
            vote_type: Type of user feedback
            
        Returns:
            bool: Success status
        """
        try:
            # Get updated state
            current_state = await self.environment.get_current_state(learner_id)
            state_after = current_state.to_feature_vector(self.environment.available_concepts)
            
            # Complete the interaction and get RL experience
            result = await self.reward_manager.complete_interaction(
                interaction_id=interaction_id,
                learner_id=learner_id,
                concept_id=concept_id,
                state_after=state_after,
                vote_type=vote_type
            )
            
            if result is None:
                logger.error(f"Failed to complete interaction {interaction_id}")
                return False
            
            reward_components, state_before, action, reward, state_after, done = result
            
            # Store experience in agent's replay buffer (if in training mode)
            if self.training_mode and self.agent:
                self.agent.remember(state_before, action, reward, state_after, done)
                
                # Trigger training if enough experiences
                if self.agent.replay_buffer.can_sample(self.agent.batch_size):
                    loss = self.agent.train()
                    if loss is not None:
                        logger.debug(f"RL training loss: {loss:.4f}")
            
            logger.info(f"Processed feedback for interaction {interaction_id}: "
                       f"reward={reward:.3f}, vote={vote_type.value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return False
    
    def enable_training_mode(self):
        """Enable online training mode"""
        self.training_mode = True
        if self.agent:
            self.agent.epsilon = max(0.1, self.agent.epsilon)  # Increase exploration
        logger.info("RL training mode enabled")
    
    def disable_training_mode(self):
        """Disable training mode (production mode)"""
        self.training_mode = False
        if self.agent:
            self.agent.epsilon = 0.01  # Minimal exploration
        logger.info("RL training mode disabled")
    
    def save_model(self):
        """Save the current RL model"""
        if self.agent and self.model_path:
            self.agent.save_model(self.model_path)
            logger.info(f"RL model saved to {self.model_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RL system statistics"""
        stats = {}
        
        if self.agent:
            stats.update(self.agent.get_stats())
        
        if self.reward_manager:
            stats.update(self.reward_manager.get_stats())
        
        stats.update({
            'initialized': self.is_initialized,
            'training_mode': self.training_mode,
            'model_path': self.model_path,
            'available_concepts': len(self.environment.available_concepts) if self.environment.available_concepts else 0
        })
        
        return stats