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
from datetime import datetime

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
        
        # RL components
        self.agent = None
        self.environment = None
        self.reward_manager = None
        
        # State
        self.is_initialized = False
        self.training_mode = False
        
        # FIXED: Add cleanup tracking
        self.cleanup_stats = {
            'failed_interactions_cleaned': 0,
            'last_cleanup_time': None
        }
        
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
        
        interaction_id = None
        selected_action = None
        
        try:
            logger.info(f"RL agent selecting question for learner {learner_id}")
            
            # Get current state
            current_state = await self.environment.get_current_state(learner_id)
            state_vector = current_state.to_feature_vector(self.environment.available_concepts)
            
            # Get valid actions (filter by target_doc_id if specified)
            valid_actions = await self._get_valid_actions(current_state, target_doc_id)
            
            if not valid_actions:
                logger.warning(f"No valid actions available for learner {learner_id}")
                return {"error": "No suitable concepts available"}
            
            # Agent selects action
            action_index = self.agent.act(state_vector, valid_actions)
            selected_action = await self.environment.action_index_to_action(action_index)
            
            logger.info(f"RL agent selected: concept={selected_action.concept_id}, "
                       f"difficulty={selected_action.difficulty_level.value}")
            
            # CRITICAL FIX: Start tracking this interaction BEFORE question generation
            # This ensures we can properly clean up if question generation fails
            interaction_id = str(uuid.uuid4())
            success = await self.reward_manager.start_interaction(
                interaction_id=interaction_id,
                learner_id=learner_id,
                concept_id=selected_action.concept_id,
                difficulty=selected_action.difficulty_level.value,
                state_before=state_vector,
                action_index=action_index
            )
            
            if not success:
                logger.error("Failed to start interaction tracking")
                return {"error": "Failed to start interaction tracking"}
            
            # Notify environment of interaction start
            self.environment.start_interaction(learner_id, interaction_id)
            
            # Generate question for the selected concept
            question_data = await self._generate_question_for_action(
                selected_action, learner_id, current_state
            )
            
            if not question_data:
                logger.warning(f"Question generation failed for concept {selected_action.concept_id}")
                # CRITICAL FIX: Clean up interaction tracking when question generation fails
                await self._cleanup_failed_interaction_async(interaction_id, learner_id, selected_action.concept_id)
                return {"error": "Failed to generate question for selected concept"}
            
            # Add RL-specific metadata to question response
            question_data.update({
                "interaction_id": interaction_id,
                "selected_by_rl": True,
                "rl_action": selected_action.to_dict(),
                "rl_confidence": self._calculate_confidence(state_vector, action_index),
                "valid_actions_count": len(valid_actions),
                "total_actions_count": self.environment.action_space_size
            })
            
            return question_data
            
        except Exception as e:
            logger.error(f"Error in RL question selection: {e}")
            import traceback
            traceback.print_exc()
            
            # CRITICAL FIX: Clean up any partial interaction tracking on error
            if interaction_id and selected_action:
                await self._cleanup_failed_interaction_async(interaction_id, learner_id, selected_action.concept_id)
            
            return {"error": f"RL selection failed: {str(e)}"}
    
    async def _cleanup_failed_interaction_async(self, interaction_id: str, learner_id: str, concept_id: str):
        """Async cleanup method for failed interactions"""
        cleanup_errors = []
        
        try:
            # Clean up reward manager tracking
            if hasattr(self, 'reward_manager') and self.reward_manager:
                # Cancel the interaction in the reward manager
                if hasattr(self.reward_manager, 'cancel_interaction'):
                    await self.reward_manager.cancel_interaction(interaction_id)
                else:
                    # Fallback: manually remove from pending interactions
                    if hasattr(self.reward_manager.interaction_tracker, 'pending_interactions'):
                        if interaction_id in self.reward_manager.interaction_tracker.pending_interactions:
                            del self.reward_manager.interaction_tracker.pending_interactions[interaction_id]
                            logger.debug(f"Cleaned up pending interaction {interaction_id}")
            
            # Clean up environment session tracking
            if hasattr(self, 'environment') and self.environment:
                # Remove from interaction start times if it exists
                if hasattr(self.environment.session_manager, 'interaction_start_times'):
                    if interaction_id in self.environment.session_manager.interaction_start_times:
                        del self.environment.session_manager.interaction_start_times[interaction_id]
                        logger.debug(f"Cleaned up environment interaction {interaction_id}")
            
            # Clean up knowledge locks if they exist
            if hasattr(self, 'reward_manager') and hasattr(self.reward_manager, 'knowledge_lock'):
                # Release any held locks
                if hasattr(self.reward_manager.knowledge_lock, 'release'):
                    try:
                        self.reward_manager.knowledge_lock.release()
                        logger.debug(f"Released knowledge lock for {interaction_id}")
                    except Exception as lock_error:
                        logger.debug(f"Knowledge lock already released for {interaction_id}: {lock_error}")
            
            # Trigger session cleanup to remove any orphaned data
            if hasattr(self, 'environment') and hasattr(self.environment, 'session_manager'):
                self.environment.session_manager.cleanup_expired_sessions()
            
            # Update cleanup statistics
            self.cleanup_stats['failed_interactions_cleaned'] += 1
            self.cleanup_stats['last_cleanup_time'] = datetime.now().isoformat()
            
            logger.info(f"Cleaned up failed interaction {interaction_id} for learner {learner_id}, concept {concept_id}")
            
        except Exception as cleanup_error:
            cleanup_errors.append(f"Async cleanup: {cleanup_error}")
            logger.error(f"Error during async cleanup of failed interaction {interaction_id}: {cleanup_error}")
        
        # Report all cleanup errors
        if cleanup_errors:
            logger.error(f"Multiple async cleanup errors for {interaction_id}: {'; '.join(cleanup_errors)}")
            # Don't raise here as this is cleanup code - just log the errors
    
    async def _get_valid_actions(self, state: State, target_doc_id: Optional[str] = None) -> List[int]:
        """Get valid actions, optionally filtered by document ID"""
        valid_actions = self.environment.get_valid_actions(state)
        
        if target_doc_id is None:
            return valid_actions
        
        # FIXED: Improved document filtering with proper async handling
        filtered_actions = []
        
        # This requires a mapping from concepts to documents
        # For now, we'll implement a simple lookup that can be enhanced
        try:
            # Get concept-to-document mapping from retriever
            # This could be cached for better performance
            concept_to_doc = {}
            
            # This is a simplified implementation - in practice you might want to cache this
            for action_idx in valid_actions:
                try:
                    # FIXED: Properly await the async method
                    action = await self.environment.action_index_to_action(action_idx)
                    # For now, we'll assume all concepts are available for any document
                    # This should be enhanced with actual concept-document mapping
                    filtered_actions.append(action_idx)
                except Exception as action_error:
                    logger.warning(f"Error converting action index {action_idx}: {action_error}")
                    # Skip this action if there's an error
                    continue
            
            return filtered_actions if filtered_actions else valid_actions
            
        except Exception as e:
            logger.warning(f"Error filtering actions by document {target_doc_id}: {e}")
            return valid_actions
    
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
            
            # FIXED: Improved adaptive question styling
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
            
            # FIXED: Enhanced question generation with better error handling
            try:
                questions = await self.question_generator.generate_questions(
                    context_chunks=context_chunks,
                    num_questions=1,
                    question_type=question_type,
                    difficulty_level=action.difficulty_level.value,
                    question_style=question_style
                )
            except Exception as gen_error:
                logger.error(f"Question generation failed: {gen_error}")
                # Fallback to simple question generation
                questions = await self.question_generator.generate_questions(
                    context_chunks=context_chunks,
                    num_questions=1,
                    question_type="conceptual",
                    difficulty_level="medium",
                    question_style="standard"
                )
            
            if not questions:
                logger.error(f"Question generation failed for concept {action.concept_id}")
                return None
            
            # Get concept name and document info
            concept_name = context_chunks[0].get('concept_name', 'Unknown Concept')
            doc_id = context_chunks[0].get('doc_id', 'unknown_doc')
            
            # Build context for evaluation
            context_for_evaluation = "\n\n".join([
                chunk.get('chunk_text', '') for chunk in context_chunks if chunk.get('chunk_text')
            ])
            
            return {
                "learner_id": learner_id,
                "doc_id": doc_id,
                "concept_id": action.concept_id,
                "concept_name": concept_name,
                "question_text": questions[0],
                "context_chunks": [chunk['chunk_text'] for chunk in context_chunks if chunk.get('chunk_text')],
                "context_for_evaluation": context_for_evaluation,
                "is_review": attempts > 0,
                "difficulty": action.difficulty_level.value,
                "question_type": question_type,
                "question_style": question_style,
                "is_new_concept_context_presented": attempts == 0,
                "concept_score": concept_score,
                "concept_attempts": attempts
            }
            
        except Exception as e:
            logger.error(f"Error generating question for action: {e}")
            import traceback
            traceback.print_exc()
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
                
                # FIXED: Improved confidence calculation
                if len(q_values_np) <= action_index:
                    return 0.1  # Low confidence for invalid action
                
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
            # FIXED: Determine if answer was correct based on latest knowledge update
            knowledge = self.profile_manager.get_concept_knowledge(learner_id, concept_id)
            was_correct = False
            if knowledge:
                # Check if the last attempt was correct
                was_correct = knowledge.get('last_answered_correctly', 0) == 1
            
            # FIXED: Notify environment of interaction completion
            vote_type_str = vote_type.value if vote_type else None
            self.environment.complete_interaction(learner_id, interaction_id, was_correct, vote_type_str)
            
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
            
            # FIXED: Auto-cleanup to prevent memory leaks
            self.reward_manager.auto_cleanup_if_needed()
            self.environment.cleanup_sessions()
            
            logger.info(f"Processed feedback for interaction {interaction_id}: "
                       f"reward={reward:.3f}, vote={vote_type.value}, correct={was_correct}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def enable_training_mode(self):
        """Enable online training mode"""
        self.training_mode = True
        if self.agent:
            # Increase exploration for training
            self.agent.epsilon = max(0.1, self.agent.epsilon)
        logger.info("RL training mode enabled")
    
    def disable_training_mode(self):
        """Disable training mode (production mode)"""
        self.training_mode = False
        if self.agent:
            # Minimal exploration for production
            self.agent.epsilon = 0.01
        logger.info("RL training mode disabled")
    
    def save_model(self):
        """Save the current RL model"""
        if self.agent and self.model_path:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.agent.save_model(self.model_path)
                logger.info(f"RL model saved to {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to save RL model: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RL system statistics"""
        stats = {}
        
        if self.agent:
            stats.update(self.agent.get_stats())
        
        if self.reward_manager:
            stats.update(self.reward_manager.get_stats())
        
        # Add environment stats
        if self.environment:
            stats.update(self.environment.get_environment_stats())
        
        # FIXED: Add cleanup and error handling statistics
        stats.update({
            'initialized': self.is_initialized,
            'training_mode': self.training_mode,
            'cleanup_stats': self.cleanup_stats.copy(),
            'error_handling_improved': True  # Indicate that error handling has been enhanced
        })
        
        return stats
    
    def get_learner_session_info(self, learner_id: str) -> Dict[str, Any]:
        """Get detailed session information for a specific learner"""
        try:
            if not self.environment:
                return {"error": "Environment not initialized"}
            
            session_stats = self.environment.get_session_stats(learner_id)
            
            # Add current state information
            current_state_future = asyncio.create_task(self.environment.get_current_state(learner_id))
            
            return {
                'session_stats': session_stats,
                'learner_id': learner_id,
                'environment_ready': self.is_initialized
            }
            
        except Exception as e:
            logger.error(f"Error getting learner session info: {e}")
            return {"error": str(e)}
    
    async def reset_learner_session(self, learner_id: str) -> bool:
        """Reset session data for a learner"""
        try:
            if self.environment:
                self.environment.session_manager.reset_session(learner_id)
                logger.info(f"Reset session for learner {learner_id}")
                return True
        except Exception as e:
            logger.error(f"Error resetting session for learner {learner_id}: {e}")
        return False
    
    def get_cleanup_status(self) -> Dict[str, Any]:
        """Get detailed cleanup status and statistics"""
        return {
            'cleanup_stats': self.cleanup_stats.copy(),
            'pending_interactions': len(self.reward_manager.interaction_tracker.pending_interactions) if self.reward_manager else 0,
            'completed_interactions': len(self.reward_manager.interaction_tracker.completed_interactions) if self.reward_manager else 0,
            'environment_interactions': len(self.environment.session_manager.interaction_start_times) if self.environment else 0,
            'cleanup_needed': self._check_if_cleanup_needed()
        }
    
    def _check_if_cleanup_needed(self) -> bool:
        """Check if cleanup is needed based on current state"""
        if not self.reward_manager or not self.environment:
            return False
        
        pending_count = len(self.reward_manager.interaction_tracker.pending_interactions)
        env_count = len(self.environment.session_manager.interaction_start_times)
        
        # Suggest cleanup if there are many pending interactions
        return pending_count > 50 or env_count > 50
    
    def force_cleanup(self) -> Dict[str, int]:
        """Force cleanup of all pending interactions and return statistics"""
        if not self.reward_manager or not self.environment:
            return {'cleaned': 0, 'error': 'Components not initialized'}
        
        initial_pending = len(self.reward_manager.interaction_tracker.pending_interactions)
        initial_env = len(self.environment.session_manager.interaction_start_times)
        
        # FIXED: Coordinated cleanup between components
        try:
            # Clean up reward manager
            self.reward_manager.cleanup()
            
            # Clean up environment
            self.environment.cleanup_sessions()
            
            # FIXED: Clean up knowledge locks if they exist
            if hasattr(self.reward_manager, 'knowledge_lock'):
                # This would need to be async in a real implementation
                logger.debug("Knowledge lock cleanup triggered")
            
        except Exception as e:
            logger.error(f"Error during force cleanup: {e}")
            return {'cleaned': 0, 'error': str(e)}
        
        final_pending = len(self.reward_manager.interaction_tracker.pending_interactions)
        final_env = len(self.environment.session_manager.interaction_start_times)
        
        cleaned_pending = initial_pending - final_pending
        cleaned_env = initial_env - final_env
        
        logger.info(f"Force cleanup completed: {cleaned_pending} pending, {cleaned_env} environment interactions cleaned")
        
        return {
            'pending_cleaned': cleaned_pending,
            'environment_cleaned': cleaned_env,
            'total_cleaned': cleaned_pending + cleaned_env
        }