# src/adaptive_engine/question_selector.py
import random
from typing import List, Dict, Optional, Any, Set
import os 

from src.learner_model.profile_manager import LearnerProfileManager
from src.retrieval.retriever import Retriever
from src.generation.question_generator_rag import RAGQuestionGenerator
from src import config 

LOW_SCORE_THRESHOLD = 5.0 
PERFECT_SCORE_THRESHOLD = 9.0 
NUM_CONTEXT_CHUNKS_FOR_NEW_CONCEPT_MAX = 2 
NUM_CONTEXT_CHUNKS_FOR_REVIEW_MAX = 3    

class QuestionSelector:
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
        
        self.curriculum_map: List[Dict[str, Any]] = [] 
        self._load_curriculum_map() 
        
        print(f"QuestionSelector initialized. Loaded {len(self.curriculum_map)} unique conceptual blocks into curriculum map.")

    def _load_curriculum_map(self):
        print("QuestionSelector: Loading/refreshing curriculum map...")
        all_chunks_meta = self.retriever.get_all_chunks_metadata(
            properties=["parent_block_id", "concept_name", "concept_type", 
                        "source_path", "original_doc_type", "doc_id", 
                        "filename", 
                        "chunk_id", "sequence_in_block"]
        )

        if not all_chunks_meta:
            print("QuestionSelector: WARNING - No chunk metadata found to build curriculum map.")
            self.curriculum_map = []
            return

        temp_map: Dict[str, Dict[str, Any]] = {}
        for chunk_meta in all_chunks_meta:
            parent_id = chunk_meta.get("parent_block_id")
            if not parent_id: continue
            
            if parent_id not in temp_map:
                doc_id_val = chunk_meta.get("doc_id")
                source_path_val = chunk_meta.get("source_path")
                filename_val = chunk_meta.get("filename")
                if not doc_id_val and source_path_val: 
                    doc_id_val = os.path.splitext(os.path.basename(source_path_val))[0]
                if not filename_val and source_path_val: 
                    filename_val = os.path.basename(source_path_val)

                temp_map[parent_id] = {
                    "concept_id": parent_id, 
                    "concept_name": chunk_meta.get("concept_name", "Unnamed Concept Block"),
                    "concept_type": chunk_meta.get("concept_type", "unknown"),
                    "source_path": source_path_val or "unknown",
                    "original_doc_type": chunk_meta.get("original_doc_type", "unknown"),
                    "doc_id": doc_id_val or "unknown_doc", 
                    "filename": filename_val or "unknown_file.ext" 
                }
        
        self.curriculum_map = list(temp_map.values())
        self.curriculum_map.sort(key=lambda x: (x.get("doc_id", ""), x.get("concept_name", "")))
        print(f"QuestionSelector: Built curriculum map with {len(self.curriculum_map)} unique conceptual blocks.")

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
            knowledge = self.profile_manager.get_concept_knowledge(learner_id, concept_id)
            if knowledge:
                score = knowledge.get("current_score", 0.0)
                attempts = knowledge.get("total_attempts", 0)
                if attempts == 0: 
                    difficulty = "beginner"
                    question_type = "factual" 
                    question_style = random.choice(["standard", "fill_in_blank"])
                elif score < LOW_SCORE_THRESHOLD: 
                    difficulty = "beginner"
                    question_type = random.choice(["factual", "conceptual"])
                    question_style = random.choice(["standard", "fill_in_blank"])
                elif score >= PERFECT_SCORE_THRESHOLD: 
                    difficulty = "advanced"
                    question_type = random.choice(["conceptual", "application", "reasoning"])
                    question_style = random.choice(["standard", "complete_proof_step"]) 
                else: 
                    difficulty = "intermediate"
                    question_type = random.choice(["conceptual", "application"])
                    question_style = "standard"
            else: 
                difficulty = "beginner"
                question_type = "factual"
                question_style = random.choice(["standard", "fill_in_blank"])
        else: 
            difficulty = "beginner"
            question_type = "factual"
            question_style = "standard"
        
        print(f"QuestionSelector: Determined params - Difficulty: {difficulty}, Type: {question_type}, Style: {question_style} for concept_id: {concept_id}")
        return {"difficulty": difficulty, "type": question_type, "style": question_style}

    async def _select_concept_for_review(self, learner_id: str, target_doc_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        print(f"QuestionSelector: Checking for concepts to review for learner {learner_id}" + (f" within topic '{target_doc_id}'." if target_doc_id else "."))
        if not self.curriculum_map: return None

        candidate_blocks = self.curriculum_map
        if target_doc_id: 
            candidate_blocks = [block for block in self.curriculum_map if block.get("doc_id") == target_doc_id]
            print(f"  Filtered to {len(candidate_blocks)} blocks for topic '{target_doc_id}'.")

        low_score_concepts = []
        for concept_block in candidate_blocks:
            concept_id = concept_block["concept_id"] 
            knowledge = self.profile_manager.get_concept_knowledge(learner_id, concept_id)
            if knowledge and knowledge.get("current_score", 10.0) < LOW_SCORE_THRESHOLD and knowledge.get("total_attempts", 0) > 0:
                concept_block_copy = concept_block.copy() 
                concept_block_copy["current_score"] = knowledge.get("current_score")
                concept_block_copy["is_review"] = True
                low_score_concepts.append(concept_block_copy)
        
        if low_score_concepts:
            low_score_concepts.sort(key=lambda x: x.get("current_score", LOW_SCORE_THRESHOLD))
            selected_for_review = low_score_concepts[0] 
            print(f"QuestionSelector: Selected concept '{selected_for_review['concept_name']}' (ID: {selected_for_review['concept_id']}) for review (score: {selected_for_review['current_score']}).")
            return selected_for_review
        
        print("QuestionSelector: No concepts found needing review based on current criteria.")
        return None

    # --- ADDED last_attempted_doc_id parameter back ---
    async def _select_new_concept(self, learner_id: str, target_doc_id: Optional[str] = None, last_attempted_doc_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
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
            knowledge = self.profile_manager.get_concept_knowledge(learner_id, concept_id)
            if not knowledge or knowledge.get("total_attempts", 0) == 0: 
                concept_block_copy = concept_block.copy()
                concept_block_copy["is_review"] = False
                potential_new_concepts.append(concept_block_copy)
            elif knowledge.get("current_score", 0.0) < PERFECT_SCORE_THRESHOLD: 
                concept_block_copy = concept_block.copy()
                concept_block_copy["is_review"] = False 
                potential_new_concepts.append(concept_block_copy)
        
        if not potential_new_concepts:
            print("QuestionSelector: No new or unmastered concepts found based on current criteria.")
            return None
        
        # Use last_attempted_doc_id for adjacency if provided
        if last_attempted_doc_id and target_doc_id is None: # Only use if not already targeting a specific doc
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
        print(f"\nQuestionSelector: Selecting next question for learner '{learner_id}'" + (f" within topic '{target_doc_id}'." if target_doc_id else "."))
        self.profile_manager.create_profile(learner_id) 

        if not self.curriculum_map: 
            print("QuestionSelector: Curriculum map is empty, attempting to reload.")
            self._load_curriculum_map()
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
            # Pass last_doc_id_for_adjacency to _select_new_concept
            selected_concept_block_info = await self._select_new_concept(learner_id, target_doc_id, last_attempted_doc_id=last_doc_id_for_adjacency)
            is_review_selection = False 

        if not selected_concept_block_info:
            msg = "Could not select any suitable concept (review or new)" + (f" for topic '{target_doc_id}'." if target_doc_id else ".")
            print(f"QuestionSelector: {msg}")
            return {"error": msg, "suggestion": "Try a different topic or broaden search."} 

        parent_block_id_for_qg = selected_concept_block_info["concept_id"] 
        concept_name_for_qg = selected_concept_block_info.get("concept_name", "N/A")
        
        q_params = await self._determine_question_params(learner_id, parent_block_id_for_qg)
        difficulty = q_params["difficulty"]
        question_type = q_params["type"]
        question_style = q_params["style"]
        
        context_limit = NUM_CONTEXT_CHUNKS_FOR_REVIEW_MAX if is_review_selection else NUM_CONTEXT_CHUNKS_FOR_NEW_CONCEPT_MAX

        print(f"QuestionSelector: Fetching context for concept_id (parent_block_id) '{parent_block_id_for_qg}' with limit {context_limit}.")
        context_chunks_data = self.retriever.get_chunks_for_parent_block(parent_block_id_for_qg, limit=context_limit)

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
            "is_new_concept_context_presented": is_new_context_presentation 
        }
