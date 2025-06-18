# src/adaptive_engine/question_selector.py
import asyncio
import random
import datetime # For getting current date for review
import logging
from typing import List, Dict, Optional, Any
import re

from src.learner_model.profile_manager import LearnerProfileManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.question_generator_rag import RAGQuestionGenerator
from src import config 
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOW_SCORE_THRESHOLD = 5.0 
PERFECT_SCORE_THRESHOLD = 9.0 
NUM_CONTEXT_CHUNKS_FOR_NEW_CONCEPT_MAX = 2 
NUM_CONTEXT_CHUNKS_FOR_REVIEW_MAX = 3    

class QuestionSelector:
    def __init__(self,
                 profile_manager: LearnerProfileManager,
                 retriever: HybridRetriever,
                 question_generator: RAGQuestionGenerator):
        
        # Removed isinstance checks to allow for easier mocking in tests.
        # The 'spec' in the mock provides sufficient interface safety.
        self.profile_manager = profile_manager
        self.retriever = retriever
        self.question_generator = question_generator
        
        # Initialize curriculum map
        self.curriculum_map = []
        
        # Initialize review thresholds
        self.review_threshold = 0.3  # Concepts below this score need review
        self.new_concept_threshold = 0.7  # Concepts above this score are considered learned

    async def initialize(self):
        """Async initializer to load the curriculum map."""
        await self._load_curriculum_map()
        logger.info("QuestionSelector initialized with %d concepts in curriculum map", 
                   len(self.curriculum_map))

    async def _load_curriculum_map(self):
        """Load the curriculum map from the retriever."""
        try:
            # Get all chunks from the retriever
            chunks = await self.retriever.get_all_chunks_metadata()
            
            # Create curriculum map entries
            for chunk in chunks:
                self.curriculum_map.append({
                    'concept_id': chunk['chunk_id'],
                    'text': chunk['text'],
                    'prerequisites': []  # TODO: Implement prerequisite detection
                })
            
            logger.info("Loaded %d concepts into curriculum map", len(self.curriculum_map))
        except Exception as e:
            logger.error("Error loading curriculum map: %s", str(e))
            self.curriculum_map = []

    def get_available_topics(self) -> List[Dict[str, str]]:
        if not self.curriculum_map: self._load_curriculum_map() 
        topics = {} 
        for block in self.curriculum_map:
            doc_id = block.get("doc_id")
            filename = block.get("filename") 
            if not filename and block.get("source_path") and block.get("source_path") != "unknown":
                filename = os.path.basename(block.get("source_path"))
            if doc_id and doc_id not in topics:
                topics[doc_id] = {"topic_id": doc_id, "source_file": filename or "Unknown Source File"}
        return list(topics.values())

    async def _determine_question_params(self, learner_id: str, concept_id: Optional[str]) -> Dict[str, Any]:
        difficulty = "intermediate"
        question_type = "conceptual" 
        question_style = "standard"  

        if concept_id:
            knowledge = await self.profile_manager.get_concept_knowledge(learner_id, concept_id)
            if knowledge:
                score = knowledge.get("current_score", 0.0)
                attempts = knowledge.get("total_attempts", 0)
                current_difficulty_level = knowledge.get("current_difficulty_level", "beginner")
                
                # Use the stored difficulty level as the base
                difficulty = current_difficulty_level

                # Adjust question type/style based on score or attempts
                if attempts == 0: # First time seeing this specific concept
                    question_type = "factual" 
                    question_style = random.choice(["standard", "fill_in_blank"])
                elif score < LOW_SCORE_THRESHOLD: 
                    question_type = random.choice(["factual", "conceptual"])
                    question_style = random.choice(["standard", "fill_in_blank"])
                elif score >= PERFECT_SCORE_THRESHOLD: 
                    # If mastered at current difficulty, next question could be harder or different style
                    question_type = random.choice(["conceptual", "application", "reasoning"])
                    question_style = random.choice(["standard", "complete_proof_step"])
                else: # Intermediate score
                    question_type = random.choice(["conceptual", "application"])
            else: # No record, treat as new
                difficulty = "beginner"
                question_type = "factual"
                question_style = random.choice(["standard", "fill_in_blank"])
        else: 
            difficulty = "beginner"
            question_type = "factual"
        
        print(f"QuestionSelector: Determined params - Difficulty: {difficulty}, Type: {question_type}, Style: {question_style} for concept_id: {concept_id}")
        return {"difficulty": difficulty, "type": question_type, "style": question_style}

    async def _select_concept_for_review(self, learner_id: str, target_doc_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        print(f"QuestionSelector: Checking for concepts to review for learner {learner_id}" + (f" within topic '{target_doc_id}'." if target_doc_id else "."))
        
        review_candidates_from_db = await self.profile_manager.get_concepts_for_review(learner_id, target_doc_id=target_doc_id)
        
        if not review_candidates_from_db:
            print("QuestionSelector: No concepts currently due for review based on SRS schedule" + (f" for topic '{target_doc_id}'." if target_doc_id else "."))
            return None

        # The review_candidates_from_db contains concept_id, score, next_review_at etc.
        # We need to find the full concept_block info from our curriculum_map.
        # The concept_id from DB is the parent_block_id.
        
        concepts_to_review_with_details = []
        for review_item in review_candidates_from_db:
            concept_id = review_item["concept_id"]
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
        # ... (logic remains same as question_selector_v2_curriculum) ...
        print(f"QuestionSelector: Selecting a new concept for learner {learner_id}" + (f" within topic '{target_doc_id}'." if target_doc_id else "."))
        if not self.curriculum_map: return None
        candidate_blocks = self.curriculum_map
        if target_doc_id: 
            candidate_blocks = [block for block in self.curriculum_map if block.get("doc_id") == target_doc_id]
            if not candidate_blocks:
                 print(f"  No conceptual blocks found for topic '{target_doc_id}' in curriculum map.")
                 return None 
            print(f"  Filtered to {len(candidate_blocks)} blocks for new concept selection in topic '{target_doc_id}'.")

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
        # ... (rest of the method uses the updated _select_concept_for_review and _determine_question_params) ...
        print(f"\nQuestionSelector: Selecting next question for learner '{learner_id}'" + (f" within topic '{target_doc_id}'." if target_doc_id else "."))
        await self.profile_manager.create_profile(learner_id) 

        if not self.curriculum_map: 
            print("QuestionSelector: Curriculum map is empty, attempting to reload.")
            await self._load_curriculum_map()
            if not self.curriculum_map:
                print("QuestionSelector: Curriculum map still empty after reload. Cannot select question.")
                return {"error": "No curriculum content available."} 

        selected_concept_block_info: Optional[Dict[str, Any]] = None 
        is_review_selection = False
        last_doc_id_for_adjacency = None

        selected_concept_block_info = await self._select_concept_for_review(learner_id, target_doc_id)
        if selected_concept_block_info:
            is_review_selection = True
            last_doc_id_for_adjacency = selected_concept_block_info.get("doc_id")

        if not selected_concept_block_info:
            selected_concept_block_info = await self._select_new_concept(learner_id, target_doc_id, last_attempted_doc_id=last_doc_id_for_adjacency)
            is_review_selection = False 

        if not selected_concept_block_info:
            msg = "Could not select any suitable concept (review or new)" + (f" for topic '{target_doc_id}'." if target_doc_id else ".")
            print(f"QuestionSelector: {msg}")
            return {"error": msg, "suggestion": "Try a different topic or broaden search."} 

        parent_block_id_for_qg = selected_concept_block_info["concept_id"] 
        concept_name_for_qg = selected_concept_block_info.get("concept_name")
        if not concept_name_for_qg:
            # Try to extract the actual definition title from the text
            context_chunks = await self.retriever.get_chunks_for_parent_block(parent_block_id_for_qg, limit=1)
            if context_chunks and context_chunks[0].get('chunk_text'):
                text = context_chunks[0]['chunk_text']
                # Look for patterns like "Definition N (Title)"
                match = re.search(r'Definition\s+\d+\s+\(([^)]+)\)', text)
                if match:
                    concept_name_for_qg = match.group(1)
                else:
                    # If no title in parentheses, try to get the first sentence
                    first_sentence = text.split('.')[0].strip()
                    if first_sentence:
                        concept_name_for_qg = first_sentence
                    else:
                        # Fallback to type and ID
                        concept_type = parent_block_id_for_qg.split('-')[0].title()
                        unique_id = parent_block_id_for_qg[-6:]
                        concept_name_for_qg = f"{concept_type} {unique_id}"
        
        q_params = await self._determine_question_params(learner_id, parent_block_id_for_qg)
        difficulty = q_params["difficulty"]
        question_type = q_params["type"]
        question_style = q_params["style"]
        
        context_limit = NUM_CONTEXT_CHUNKS_FOR_REVIEW_MAX if is_review_selection else NUM_CONTEXT_CHUNKS_FOR_NEW_CONCEPT_MAX

        print(f"QuestionSelector: Fetching context for concept_id (parent_block_id) '{parent_block_id_for_qg}' with limit {context_limit}.")
        context_chunks_data = await self.retriever.get_chunks_for_parent_block(parent_block_id_for_qg, limit=context_limit)

        if not context_chunks_data:
            msg = f"No context chunks found by retriever for parent_block_id '{parent_block_id_for_qg}'."
            print(f"QuestionSelector: {msg}")
            return {"error": msg} 
        
        full_context_for_qg = "\n\n".join([chk.get("chunk_text", "") for chk in context_chunks_data if chk.get("chunk_text","").strip()])
        
        if not full_context_for_qg.strip():
            msg = f"Context for parent_block_id '{parent_block_id_for_qg}' is empty after concatenating chunks."
            print(f"QuestionSelector: {msg}")
            return {"error": msg}

        is_new_context_presentation = (not is_review_selection and difficulty in ["beginner", "intermediate"])
        if is_new_context_presentation:
            print("\n--- Context for New Concept ---")
            print(f"Topic: {concept_name_for_qg}")
            print("Please review the following information before answering the question:")
            print("--------------------------------------------------------------------")
            print(full_context_for_qg)
            print("--------------------------------------------------------------------")

        print(f"QuestionSelector: Generating {difficulty} '{question_type}' question (style: {question_style}) for concept '{concept_name_for_qg}'.")
        
        context_list_for_generator = [{"chunk_text": full_context_for_qg}]
        
        generated_questions = await self.question_generator.generate_questions(
            context_chunks=context_list_for_generator,
            num_questions=1, 
            question_type=question_type, 
            difficulty_level=difficulty,
            question_style=question_style
        )

        if not generated_questions:
            msg = f"Failed to generate question for concept '{concept_name_for_qg}'."
            print(f"QuestionSelector: {msg}")
            return {"error": msg}

        question_text = generated_questions[0]
        
        print(f"QuestionSelector: Selected question: \"{question_text}\"")
        return {
            "concept_id": parent_block_id_for_qg, 
            "concept_name": concept_name_for_qg,
            "question_text": question_text,
            "context_for_evaluation": full_context_for_qg,
            "is_new_concept_context_presented": is_new_context_presentation,
            "question_type": question_type,
            "difficulty": difficulty,
            "style": question_style
        }
