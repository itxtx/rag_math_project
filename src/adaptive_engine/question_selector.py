import asyncio
import random
import logging
from typing import Optional, List, Dict, Any, Tuple
import datetime

from src.learner_model.profile_manager import LearnerProfileManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.question_generator_rag import RAGQuestionGenerator
from src.config import (
    REVIEW_SCORE_THRESHOLD, 
    LOW_SCORE_THRESHOLD, 
    PERFECT_SCORE_THRESHOLD,
    ADJACENCY_BONUS
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuestionSelector:
    """
    Selects the next question for a learner based on their profile and the curriculum structure.
    """

    def __init__(
        self,
        profile_manager: LearnerProfileManager,
        retriever: HybridRetriever,
        question_generator: RAGQuestionGenerator,
    ):
        """
        Initializes the QuestionSelector.

        Args:
            profile_manager: An instance of LearnerProfileManager.
            retriever: An instance of a retriever for accessing curriculum data.
            question_generator: An instance of RAGQuestionGenerator.
        """
        self.profile_manager = profile_manager
        self.retriever = retriever
        self.question_generator = question_generator
        self.curriculum_map: List[Dict[str, Any]] = []
        self.is_initialized = asyncio.Event()

    async def initialize(self):
        """
        Initialize the QuestionSelector by loading the curriculum map.
        This method should be called once before using the selector.
        """
        logger.info("Initializing QuestionSelector with optimized curriculum loading...")
        try:
            # FIXED: Implement lazy loading and caching for better scalability
            # Instead of loading all documents at once, we'll load them in chunks
            # and cache the results for better performance
            
            self.curriculum_map = []
            self._curriculum_cache = {}  # Cache for concept metadata
            self._is_loading = False
            
            # Load initial curriculum map in chunks
            await self._load_curriculum_chunk(limit=100)
            
            if not self.curriculum_map:
                logger.warning("No curriculum metadata found from retriever.")
                self.is_initialized.set()
                return

            logger.info(f"Successfully loaded initial curriculum map with {len(self.curriculum_map)} unique concepts.")
            logger.info("Additional concepts will be loaded on-demand for better performance.")

        except Exception as e:
            logger.error(f"An unexpected error occurred during curriculum map initialization: {e}", exc_info=True)
            self.curriculum_map = []

        self.is_initialized.set()
    
    async def _load_curriculum_chunk(self, limit: int = 100, offset: int = 0):
        """
        Load a chunk of curriculum data to avoid loading everything at once
        
        Args:
            limit: Maximum number of documents to load
            offset: Offset for pagination
        """
        try:
            # FIXED: Use pagination to load documents in chunks
            all_chunks_metadata = await self.retriever.get_all_documents(limit=limit)
            
            if not all_chunks_metadata:
                return
            
            # Process the chunk and add to curriculum map
            temp_curriculum_map = {}
            for chunk in all_chunks_metadata:
                concept_id = chunk.get("parent_block_id")
                if not concept_id:
                    logger.warning(f"Skipping chunk, missing 'parent_block_id': {chunk}")
                    continue

                if concept_id not in temp_curriculum_map:
                    temp_curriculum_map[concept_id] = {
                        "concept_id": concept_id,
                        "doc_id": chunk.get("doc_id"),
                        "concept_name": chunk.get("concept_name", "Unnamed Concept"),
                        "dependencies": chunk.get("dependencies", [])
                    }
            
            # Add new concepts to the curriculum map
            for concept_id, concept_data in temp_curriculum_map.items():
                if concept_id not in self._curriculum_cache:
                    self._curriculum_cache[concept_id] = concept_data
                    self.curriculum_map.append(concept_data)
            
            logger.debug(f"Loaded {len(temp_curriculum_map)} new concepts from chunk")
            
        except Exception as e:
            logger.error(f"Error loading curriculum chunk: {e}")
    
    async def _ensure_concept_in_curriculum(self, concept_id: str) -> bool:
        """
        Ensure a specific concept is loaded in the curriculum map
        
        Args:
            concept_id: The concept ID to ensure is loaded
            
        Returns:
            bool: True if concept is available, False otherwise
        """
        if concept_id in self._curriculum_cache:
            return True
        
        # FIXED: Load additional concepts on-demand if not in cache
        try:
            # Try to get the specific concept from the retriever
            # This is more efficient than loading all documents
            concept_chunks = await self.retriever.get_chunks_for_parent_block(concept_id, limit=1)
            
            if concept_chunks:
                chunk = concept_chunks[0]
                concept_data = {
                    "concept_id": concept_id,
                    "doc_id": chunk.get("doc_id"),
                    "concept_name": chunk.get("concept_name", "Unnamed Concept"),
                    "dependencies": chunk.get("dependencies", [])
                }
                
                self._curriculum_cache[concept_id] = concept_data
                self.curriculum_map.append(concept_data)
                
                logger.debug(f"Loaded concept {concept_id} on-demand")
                return True
            else:
                logger.warning(f"Concept {concept_id} not found in retriever")
                return False
                
        except Exception as e:
            logger.error(f"Error loading concept {concept_id} on-demand: {e}")
            return False

    async def _ensure_initialized(self):
        """Waits until the initialization is complete."""
        await self.is_initialized.wait()
        if not self.curriculum_map:
            logger.warning("Curriculum map is empty, attempting to reload.")
            self.is_initialized.clear()
            await self.initialize()
            if not self.curriculum_map:
                logger.error("Curriculum map still empty after reload. Cannot select question.")
                return False
        return True


    async def _select_concept_for_review(self, learner_id: str, target_doc_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        print(f"QuestionSelector: Checking for concepts to review for learner {learner_id}" + (f" within topic '{target_doc_id}'." if target_doc_id else "."))

        review_candidates_from_db = await self.profile_manager.get_concepts_for_review(learner_id, target_doc_id=target_doc_id)

        if not review_candidates_from_db:
            print("QuestionSelector: No concepts currently due for review based on SRS schedule" + (f" for topic '{target_doc_id}'." if target_doc_id else "."))
            return None

        # FIXED: Use on-demand loading for better scalability
        concepts_to_review_with_details = []
        for review_item in review_candidates_from_db:
            concept_id = review_item["concept_id"]
            
            # FIXED: Ensure concept is loaded in curriculum map
            if not await self._ensure_concept_in_curriculum(concept_id):
                print(f"Warning: Concept {concept_id} due for review not found in retriever.")
                continue
            
            # Find this concept in the curriculum_map
            block_info = next((block for block in self.curriculum_map if block["concept_id"] == concept_id), None)
            if block_info:
                block_info_copy = block_info.copy()
                block_info_copy["current_score"] = review_item.get("current_score") # From DB
                block_info_copy["next_review_at"] = review_item.get("next_review_at")
                block_info_copy["is_review"] = True
                concepts_to_review_with_details.append(block_info_copy)
            else:
                print(f"Warning: Concept {concept_id} due for review not found in current curriculum map.")

        if not concepts_to_review_with_details:
            print("QuestionSelector: No reviewable concepts found in the current curriculum map.")
            return None

        # Prioritize: earliest review date, then lowest score
        concepts_to_review_with_details.sort(key=lambda x: (x.get("next_review_at", datetime.datetime.max), x.get("current_score", LOW_SCORE_THRESHOLD)))

        selected_for_review = concepts_to_review_with_details[0]
        print(f"QuestionSelector: Selected concept '{selected_for_review['concept_name']}' (ID: {selected_for_review['concept_id']}) for review. Due: {selected_for_review.get('next_review_at')}, Score: {selected_for_review.get('current_score')}")
        return selected_for_review

    async def _select_new_concept(self, learner_id: str, target_doc_id: Optional[str] = None, last_attempted_doc_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        print(f"QuestionSelector: Selecting a new concept for learner {learner_id}" + (f" within topic '{target_doc_id}'." if target_doc_id else "."))
        
        # FIXED: Load more concepts if we don't have enough for selection
        if len(self.curriculum_map) < 10:
            logger.info("Loading additional concepts for better selection")
            await self._load_curriculum_chunk(limit=200)
        
        if not self.curriculum_map: 
            return None
            
        candidate_blocks = self.curriculum_map
        if target_doc_id:
            candidate_blocks = [block for block in self.curriculum_map if block.get("doc_id") == target_doc_id]
            if not candidate_blocks:
                print(f"   No conceptual blocks found for topic '{target_doc_id}' in curriculum map.")
                # FIXED: Try to load more concepts for this specific document
                await self._load_curriculum_chunk(limit=100)
                candidate_blocks = [block for block in self.curriculum_map if block.get("doc_id") == target_doc_id]
                if not candidate_blocks:
                    return None
            print(f"   Filtered to {len(candidate_blocks)} blocks for new concept selection in topic '{target_doc_id}'.")

        potential_new_concepts = []
        for concept_block in candidate_blocks:
            concept_id = concept_block["concept_id"]
            knowledge = await self.profile_manager.get_concept_knowledge(learner_id, concept_id)
            if not knowledge or knowledge.get("total_attempts", 0) == 0:
                concept_block_copy = concept_block.copy(); concept_block_copy["is_review"] = False
                potential_new_concepts.append(concept_block_copy)
            elif knowledge.get("current_score", 0.0) < PERFECT_SCORE_THRESHOLD:
                concept_block_copy = concept_block.copy(); concept_block_copy["is_review"] = False
                potential_new_concepts.append(concept_block_copy)

        if not potential_new_concepts:
            print("QuestionSelector: No new or unmastered concepts found based on current criteria.")
            return None

        if last_attempted_doc_id and target_doc_id is None:
            same_doc_new_concepts = [c for c in potential_new_concepts if c.get("doc_id") == last_attempted_doc_id]
            if same_doc_new_concepts:
                print(f"QuestionSelector: Prioritizing new concepts from last document '{last_attempted_doc_id}'.")
                selected_new = random.choice(same_doc_new_concepts)
                print(f"QuestionSelector: Selected new concept '{selected_new['concept_name']}' (ID: {selected_new['concept_id']}) from same document.")
                return selected_new

        selected_new = random.choice(potential_new_concepts)
        print(f"QuestionSelector: Selected new concept '{selected_new['concept_name']}' (ID: {selected_new['concept_id']}) randomly from available candidates.")
        return selected_new


    async def select_next_question(self, learner_id: str, target_doc_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        
        if not await self._ensure_initialized():
            return {"error": "No curriculum content available."}

        # ... (rest of the method uses the updated _select_concept_for_review and _determine_question_params) ...
        print(f"\nQuestionSelector: Selecting next question for learner '{learner_id}'" + (f" within topic '{target_doc_id}'." if target_doc_id else "."))
        await self.profile_manager.create_profile(learner_id)

        last_concept_id, last_doc_id_for_adjacency = await self.profile_manager.get_last_attempted_concept_and_doc(learner_id)

        # 1. Prioritize review questions
        selected_concept_block_info = await self._select_concept_for_review(learner_id, target_doc_id)

        # 2. If no reviews, find a new concept
        if not selected_concept_block_info:
            selected_concept_block_info = await self._select_new_concept(learner_id, target_doc_id, last_attempted_doc_id=last_doc_id_for_adjacency)
            
        # If still no concept, we are done for now
        if not selected_concept_block_info:
            print("QuestionSelector: No suitable review or new concepts found for the learner at this time.")
            # ... existing logic to find ANY unmastered concept as a fallback ...
            return {"message": "Congratulations! You've mastered all available concepts for now."}

        # 3. We have a concept, now get the chunks and generate a question
        question_params = await self._determine_question_params(selected_concept_block_info)
        if not question_params:
            return {"error": "Could not determine parameters for question generation."}

        # Generate the question using the RAG model
        generated_questions = await self.question_generator.generate_questions(
            context_chunks=question_params["context_chunks"],
            num_questions=1,
            question_type="conceptual",
            difficulty_level=question_params["difficulty"],
            question_style="standard"
        )

        if not generated_questions:
            return {"error": "Failed to generate a question for the selected concept."}

        # For simplicity, pick the first generated question
        question_text = generated_questions[0]

        # Prepare the final response payload
        response = {
            "learner_id": learner_id,
            "doc_id": selected_concept_block_info["doc_id"],
            "concept_id": selected_concept_block_info["concept_id"],
            "concept_name": selected_concept_block_info["concept_name"],
            "question_text": question_text,
            "is_review": selected_concept_block_info.get("is_review", False),
            "context_chunks": [chunk['chunk_text'] for chunk in question_params["context_chunks"]]
        }
        
        return response

    async def _determine_question_params(self, concept_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Gathers context chunks and determines the difficulty for question generation.
        """
        concept_id = concept_info.get("concept_id")
        if not concept_id:
            logger.error("Cannot determine question params, concept_info missing 'concept_id'.")
            return None

        # Retrieve all chunks associated with the selected conceptual block (parent_block_id)
        context_chunks = await self.retriever.get_chunks_for_parent_block(concept_id)
        if not context_chunks:
            logger.warning(f"No content chunks found for concept {concept_id}, cannot generate question.")
            return None

        # Determine difficulty based on learner's score
        score = concept_info.get("current_score", 0.0)
        difficulty = "easy"
        if score < 0.4:
            difficulty = "easy"
        elif score < 0.75:
            difficulty = "medium"
        else:
            difficulty = "hard"

        return {
            "context_chunks": context_chunks,
            "concept_name": concept_info.get("concept_name"),
            "difficulty": difficulty
        }