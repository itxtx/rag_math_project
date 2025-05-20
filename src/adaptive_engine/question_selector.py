# src/adaptive_engine/question_selector.py
import random
from typing import List, Dict, Optional, Any, Set
import os # For os.path.exists in demo

from src.learner_model.profile_manager import LearnerProfileManager
from src.retrieval.retriever import Retriever
from src.generation.question_generator_rag import RAGQuestionGenerator
from src import config 

# Constants for selection strategy
LOW_SCORE_THRESHOLD = 5.0 
PERFECT_SCORE_THRESHOLD = 9.0 
# NUM_CANDIDATE_CONCEPTS_TO_FETCH is less relevant if we have a full curriculum map
NUM_CONTEXT_CHUNKS_FOR_NEW_CONCEPT_MAX = 3 # Max chunks to form context for a new concept
NUM_CONTEXT_CHUNKS_FOR_REVIEW_MAX = 3    # Max chunks for review context

class QuestionSelector:
    """
    Selects or generates the next question for the learner based on their profile,
    available concepts (curriculum map), considering difficulty and comprehensible input.
    """

    def __init__(self,
                 profile_manager: LearnerProfileManager,
                 retriever: Retriever,
                 question_generator: RAGQuestionGenerator):
        if not isinstance(profile_manager, LearnerProfileManager):
            raise TypeError("profile_manager must be an instance of LearnerProfileManager.")
        if not isinstance(retriever, Retriever):
            raise TypeError("retriever must be an instance of Retriever.")
        if not isinstance(question_generator, RAGQuestionGenerator):
            raise TypeError("question_generator must be an instance of RAGQuestionGenerator.")

        self.profile_manager = profile_manager
        self.retriever = retriever
        self.question_generator = question_generator
        
        self.curriculum_map: List[Dict[str, Any]] = [] # Stores unique conceptual blocks
        self._load_curriculum_map()
        
        print(f"QuestionSelector initialized. Loaded {len(self.curriculum_map)} unique conceptual blocks into curriculum map.")

    def _load_curriculum_map(self):
        """
        Loads metadata for all unique conceptual blocks from Weaviate to build
        an in-memory curriculum map. A conceptual block is identified by 'parent_block_id'.
        """
        print("QuestionSelector: Loading curriculum map...")
        # Properties needed to identify and describe a conceptual block
        # 'parent_block_id' is the key. We also want its name, type, source.
        # These typically come from the chunk level, so we aggregate.
        # We assume that all chunks with the same parent_block_id share the same
        # concept_name, concept_type, source_path, original_type.
        
        # get_all_chunks_metadata retrieves chunk-level data.
        # We need to distill this into unique conceptual blocks.
        all_chunks_meta = self.retriever.get_all_chunks_metadata(
            properties=["parent_block_id", "concept_name", "concept_type", 
                        "source_path", "original_type", "doc_id"] # Added doc_id
        )

        if not all_chunks_meta:
            print("QuestionSelector: WARNING - No chunk metadata found to build curriculum map.")
            self.curriculum_map = []
            return

        temp_map: Dict[str, Dict[str, Any]] = {}
        for chunk_meta in all_chunks_meta:
            parent_id = chunk_meta.get("parent_block_id")
            if not parent_id: # Skip chunks without a parent_block_id
                continue
            
            if parent_id not in temp_map:
                temp_map[parent_id] = {
                    "concept_id": parent_id, # This is the ID of the conceptual block
                    "concept_name": chunk_meta.get("concept_name", "Unnamed Concept Block"),
                    "concept_type": chunk_meta.get("concept_type", "unknown"),
                    "source_path": chunk_meta.get("source_path", "unknown"),
                    "original_type": chunk_meta.get("original_type", "unknown"),
                    "doc_id": chunk_meta.get("doc_id", "unknown") # Document it belongs to
                    # We could add block_order here if it was stored with chunks or conceptual blocks
                }
        
        self.curriculum_map = list(temp_map.values())
        # Optionally sort the curriculum_map, e.g., by source_path, then by some internal order if available
        # For now, the order is based on first encounter.
        print(f"QuestionSelector: Built curriculum map with {len(self.curriculum_map)} unique conceptual blocks.")


    async def _determine_difficulty(self, 
                                   learner_id: str, 
                                   concept_id: Optional[str] # This is the parent_block_id
                                   ) -> str:
        """Determines appropriate difficulty level."""
        difficulty = "intermediate" 
        if concept_id:
            knowledge = self.profile_manager.get_concept_knowledge(learner_id, concept_id)
            if knowledge:
                score = knowledge.get("current_score", 0.0)
                attempts = knowledge.get("total_attempts", 0)
                if attempts == 0: # First time seeing this specific concept
                    difficulty = "beginner"
                elif score < LOW_SCORE_THRESHOLD:
                    difficulty = "beginner" 
                elif score >= PERFECT_SCORE_THRESHOLD:
                    difficulty = "advanced" 
            else: 
                difficulty = "beginner" # No record, treat as new
        else: 
            difficulty = "beginner"
        
        print(f"QuestionSelector: Determined difficulty: {difficulty} for concept_id: {concept_id}")
        return difficulty

    async def _select_concept_for_review(self, learner_id: str) -> Optional[Dict[str, Any]]:
        print(f"QuestionSelector: Checking for concepts to review for learner {learner_id}.")
        if not self.curriculum_map:
            print("QuestionSelector: Curriculum map is empty, cannot select concept for review.")
            return None

        low_score_concepts = []
        for concept_block in self.curriculum_map:
            concept_id = concept_block["concept_id"] # This is the parent_block_id
            knowledge = self.profile_manager.get_concept_knowledge(learner_id, concept_id)
            if knowledge and knowledge.get("current_score", 10.0) < LOW_SCORE_THRESHOLD and knowledge.get("total_attempts", 0) > 0:
                concept_block_copy = concept_block.copy() # Avoid modifying original map item
                concept_block_copy["current_score"] = knowledge.get("current_score")
                concept_block_copy["is_review"] = True
                low_score_concepts.append(concept_block_copy)
        
        if low_score_concepts:
            # Prioritize by lowest score, then perhaps recency (not tracked yet)
            low_score_concepts.sort(key=lambda x: x.get("current_score", LOW_SCORE_THRESHOLD))
            selected_for_review = low_score_concepts[0] # Pick the one with the lowest score
            print(f"QuestionSelector: Selected concept '{selected_for_review['concept_name']}' (ID: {selected_for_review['concept_id']}) for review (score: {selected_for_review['current_score']}).")
            return selected_for_review
        
        print("QuestionSelector: No concepts found needing review based on low scores.")
        return None

    async def _select_new_concept(self, learner_id: str, last_attempted_doc_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        print(f"QuestionSelector: Selecting a new concept for learner {learner_id}.")
        if not self.curriculum_map:
            print("QuestionSelector: Curriculum map is empty, cannot select new concept.")
            return None

        potential_new_concepts = []
        for concept_block in self.curriculum_map:
            concept_id = concept_block["concept_id"]
            knowledge = self.profile_manager.get_concept_knowledge(learner_id, concept_id)
            if not knowledge or knowledge.get("total_attempts", 0) == 0: # Truly new
                concept_block_copy = concept_block.copy()
                concept_block_copy["is_review"] = False
                potential_new_concepts.append(concept_block_copy)
            elif knowledge.get("current_score", 0.0) < PERFECT_SCORE_THRESHOLD: # Attempted but not mastered
                concept_block_copy = concept_block.copy()
                concept_block_copy["is_review"] = False # Treat as new if not explicitly for review
                potential_new_concepts.append(concept_block_copy)
        
        if not potential_new_concepts:
            print("QuestionSelector: No new or unmastered concepts found in curriculum map.")
            return None

        # Simplified Adjacency: Prefer concepts from the same document as last attempted, if available
        if last_attempted_doc_id:
            same_doc_new_concepts = [c for c in potential_new_concepts if c.get("doc_id") == last_attempted_doc_id]
            if same_doc_new_concepts:
                print(f"QuestionSelector: Prioritizing new concepts from last document '{last_attempted_doc_id}'.")
                selected_new = random.choice(same_doc_new_concepts)
                print(f"QuestionSelector: Selected new concept '{selected_new['concept_name']}' (ID: {selected_new['concept_id']}) from same document.")
                return selected_new
        
        # If no same-doc concepts or no last_attempted_doc_id, pick randomly from all potential new ones
        selected_new = random.choice(potential_new_concepts) 
        print(f"QuestionSelector: Selected new concept '{selected_new['concept_name']}' (ID: {selected_new['concept_id']}) randomly.")
        return selected_new


    async def select_next_question(self, learner_id: str) -> Optional[Dict[str, Any]]:
        print(f"\nQuestionSelector: Selecting next question for learner '{learner_id}'...")
        self.profile_manager.create_profile(learner_id) 

        selected_concept_block_info: Optional[Dict[str, Any]] = None # This will hold the chosen conceptual block's metadata
        is_review_selection = False

        # TODO: Integrate Spaced Repetition System (Phase 5) to identify concepts for review.
        # For now, simple low-score review:
        selected_concept_block_info = await self._select_concept_for_review(learner_id)
        if selected_concept_block_info:
            is_review_selection = True

        last_doc_id_for_new_selection = None
        if selected_concept_block_info: # If a review item was selected, use its doc_id for adjacency
            last_doc_id_for_new_selection = selected_concept_block_info.get("doc_id")

        if not selected_concept_block_info:
            selected_concept_block_info = await self._select_new_concept(learner_id, last_doc_id_for_new_selection)
            is_review_selection = False 

        if not selected_concept_block_info:
            print("QuestionSelector: Could not select any suitable concept (review or new).")
            return None

        parent_block_id_for_qg = selected_concept_block_info["concept_id"] # This is the ID of the conceptual block
        concept_name_for_qg = selected_concept_block_info.get("concept_name", "N/A")
        
        difficulty = await self._determine_difficulty(learner_id, parent_block_id_for_qg)
        context_limit = NUM_CONTEXT_CHUNKS_FOR_REVIEW_MAX if is_review_selection else NUM_CONTEXT_CHUNKS_FOR_NEW_CONCEPT_MAX

        # Get all chunks for this conceptual block to form the context
        print(f"QuestionSelector: Fetching context for concept_id (parent_block_id) '{parent_block_id_for_qg}' with limit {context_limit}.")
        context_chunks_data = self.retriever.get_chunks_for_parent_block(parent_block_id_for_qg, limit=context_limit)

        if not context_chunks_data:
            print(f"QuestionSelector: No context chunks found by retriever for parent_block_id '{parent_block_id_for_qg}'. Cannot generate question.")
            return None
        
        # Concatenate text from all retrieved chunks for this conceptual block
        full_context_for_qg = "\n\n".join([chk.get("chunk_text", "") for chk in context_chunks_data if chk.get("chunk_text","").strip()])
        
        if not full_context_for_qg.strip():
            print(f"QuestionSelector: Context for parent_block_id '{parent_block_id_for_qg}' is empty after concatenating chunks.")
            return None

        print(f"QuestionSelector: Generating {difficulty} question for concept '{concept_name_for_qg}' (ID: {parent_block_id_for_qg}).")
        
        # RAGQuestionGenerator expects a list of dicts, each with "chunk_text"
        # Here, we are providing the combined context as a single "chunk" to the generator.
        # Alternatively, pass context_chunks_data directly if QG can handle multiple small dicts.
        context_list_for_generator = [{"chunk_text": full_context_for_qg}]
        
        generated_questions = await self.question_generator.generate_questions(
            context_chunks=context_list_for_generator,
            num_questions=1, 
            question_type="conceptual", # Can be made adaptive
            difficulty_level=difficulty 
        )

        if not generated_questions:
            print(f"QuestionSelector: Failed to generate question for concept '{concept_name_for_qg}'.")
            return None

        question_text = generated_questions[0]
        
        print(f"QuestionSelector: Selected question: \"{question_text}\"")
        return {
            "concept_id": parent_block_id_for_qg, 
            "concept_name": concept_name_for_qg,
            "question_text": question_text,
            "context_for_evaluation": full_context_for_qg 
        }

