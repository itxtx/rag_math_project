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
NUM_CANDIDATE_CONCEPTS_TO_FETCH = 5 
# For comprehensible input: number of context chunks to provide for question generation
NUM_CONTEXT_CHUNKS_FOR_NEW_CONCEPT = 1 # Start with minimal context for new concepts
NUM_CONTEXT_CHUNKS_FOR_REVIEW = 2 # Maybe slightly more for review

class QuestionSelector:
    """
    Selects or generates the next question for the learner based on their profile
    and available concepts, considering difficulty and comprehensible input.
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
        print("QuestionSelector initialized.")

    async def _determine_difficulty_and_context_limit(self, 
                                                       learner_id: str, 
                                                       concept_id: Optional[str],
                                                       is_review: bool
                                                       ) -> tuple[str, int]:
        """
        Determines appropriate difficulty level and context chunk limit.
        """
        difficulty = "intermediate" # Default
        context_chunk_limit = NUM_CONTEXT_CHUNKS_FOR_NEW_CONCEPT

        if concept_id: # If we have a specific concept
            knowledge = self.profile_manager.get_concept_knowledge(learner_id, concept_id)
            if knowledge:
                score = knowledge.get("current_score", 0.0)
                attempts = knowledge.get("total_attempts", 0)

                if is_review or (attempts > 0 and score < LOW_SCORE_THRESHOLD):
                    difficulty = "beginner" # Review low scores with easier questions
                    context_chunk_limit = NUM_CONTEXT_CHUNKS_FOR_REVIEW
                elif score >= PERFECT_SCORE_THRESHOLD:
                    difficulty = "advanced" # Challenge mastered concepts
                    context_chunk_limit = NUM_CONTEXT_CHUNKS_FOR_REVIEW # Can handle more context
                elif score >= LOW_SCORE_THRESHOLD:
                    difficulty = "intermediate"
                    context_chunk_limit = NUM_CONTEXT_CHUNKS_FOR_REVIEW
                # else (very low score on first few attempts) remains beginner/intermediate
            else: # No prior knowledge for this specific concept_id
                difficulty = "beginner"
                context_chunk_limit = NUM_CONTEXT_CHUNKS_FOR_NEW_CONCEPT
        else: # No specific concept_id (e.g., selecting a brand new one generally)
            difficulty = "beginner" # Start new topics easy
            context_chunk_limit = NUM_CONTEXT_CHUNKS_FOR_NEW_CONCEPT
        
        print(f"QuestionSelector: Determined difficulty: {difficulty}, context_limit: {context_chunk_limit} for concept_id: {concept_id}")
        return difficulty, context_chunk_limit


    async def _select_concept_for_review(self, learner_id: str) -> Optional[Dict[str, Any]]:
        print(f"QuestionSelector: Checking for concepts to review for learner {learner_id}.")
        candidate_chunks = self.retriever.search(
            query_text="concepts previously studied", # Broader query, or could be targeted
            search_type="semantic", 
            limit=NUM_CANDIDATE_CONCEPTS_TO_FETCH * 3 # Fetch more to find low-score ones
        )

        if not candidate_chunks:
            print("QuestionSelector: No candidate chunks found for review check.")
            return None

        low_score_concepts = []
        for chunk in candidate_chunks:
            concept_id = chunk.get("_id") or chunk.get("chunk_id")
            if not concept_id: continue

            knowledge = self.profile_manager.get_concept_knowledge(learner_id, concept_id)
            if knowledge and knowledge.get("current_score", 10.0) < LOW_SCORE_THRESHOLD and knowledge.get("total_attempts", 0) > 0:
                # We need enough context for the question generator
                # The chunk itself is one piece of context. If retriever can give more related, that's better.
                # For now, we'll use this chunk's text as the primary context.
                if chunk.get("chunk_text"):
                    low_score_concepts.append({
                        "concept_id": concept_id,
                        "concept_name": chunk.get("concept_name", concept_id),
                        "current_score": knowledge.get("current_score"),
                        "context_text": chunk.get("chunk_text"), 
                        "source_chunk": chunk,
                        "is_review": True
                    })
        
        if low_score_concepts:
            selected_for_review = random.choice(low_score_concepts)
            print(f"QuestionSelector: Selected concept '{selected_for_review['concept_name']}' for review (score: {selected_for_review['current_score']}).")
            return selected_for_review
        
        print("QuestionSelector: No concepts found needing review based on current low score strategy.")
        return None

    async def _select_new_concept(self, learner_id: str) -> Optional[Dict[str, Any]]:
        print(f"QuestionSelector: Selecting a new concept for learner {learner_id}.")
        
        # To ensure comprehensible input for a *new* concept, the query should ideally
        # be targeted if we have a curriculum. For now, a general query.
        # The 'limit' here controls how many *different* potential concepts we look at.
        candidate_chunks = self.retriever.search(
            query_text=getattr(config, 'DEFAULT_NEW_CONCEPT_QUERY', "introduction to core principles"), 
            search_type="semantic", 
            limit=NUM_CANDIDATE_CONCEPTS_TO_FETCH
        )

        if not candidate_chunks:
            print("QuestionSelector: No candidate chunks found for new concept selection.")
            return None

        potential_new_concepts = []
        for chunk in candidate_chunks:
            concept_id = chunk.get("_id") or chunk.get("chunk_id")
            if not concept_id: continue

            knowledge = self.profile_manager.get_concept_knowledge(learner_id, concept_id)
            if (not knowledge or knowledge.get("total_attempts", 0) == 0) and chunk.get("chunk_text"):
                # Truly new (no attempts) or not yet mastered
                 potential_new_concepts.append({
                    "concept_id": concept_id,
                    "concept_name": chunk.get("concept_name", concept_id),
                    "context_text": chunk.get("chunk_text"), # This is the primary context
                    "source_chunk": chunk,
                    "is_review": False
                })
            elif knowledge and knowledge.get("current_score", 0.0) < PERFECT_SCORE_THRESHOLD and chunk.get("chunk_text"):
                # Attempted but not mastered, also a candidate for "new" if no truly new ones
                 potential_new_concepts.append({
                    "concept_id": concept_id,
                    "concept_name": chunk.get("concept_name", concept_id),
                    "context_text": chunk.get("chunk_text"),
                    "source_chunk": chunk,
                    "is_review": False # Treat as new for difficulty setting if not explicitly review
                })

        if potential_new_concepts:
            selected_new = random.choice(potential_new_concepts) 
            print(f"QuestionSelector: Selected new concept '{selected_new['concept_name']}'.")
            return selected_new
            
        print("QuestionSelector: Could not find a suitable new concept.")
        return None


    async def select_next_question(self, learner_id: str) -> Optional[Dict[str, Any]]:
        print(f"\nQuestionSelector: Selecting next question for learner '{learner_id}'...")
        self.profile_manager.create_profile(learner_id) 

        selected_concept_info = None
        is_review_selection = False

        selected_concept_info = await self._select_concept_for_review(learner_id)
        if selected_concept_info:
            is_review_selection = True

        if not selected_concept_info:
            selected_concept_info = await self._select_new_concept(learner_id)
            is_review_selection = False # It's a new concept

        if not selected_concept_info or not selected_concept_info.get("context_text"):
            print("QuestionSelector: Could not select a suitable concept or context is missing.")
            return None

        concept_id = selected_concept_info["concept_id"]
        context_for_qg_text = selected_concept_info["context_text"] # This is the single chunk text
        concept_name = selected_concept_info.get("concept_name", "N/A")
        
        # Determine difficulty and how much context to actually use
        difficulty, context_chunk_limit = await self._determine_difficulty_and_context_limit(
            learner_id, concept_id, is_review_selection
        )

        # For comprehensible input, we might want to fetch more focused context around this specific concept_id
        # The current selected_concept_info.context_text is from a single retrieved chunk.
        # If context_chunk_limit > 1, we could try to get more related chunks.
        # For now, we'll use the context from the selected chunk.
        # To expand context:
        # final_context_chunks_for_qg = [selected_concept_info["source_chunk"]]
        # if context_chunk_limit > 1 and concept_name != "N/A":
        #     # Try to get more chunks specifically about this concept_name or concept_id
        #     more_chunks = self.retriever.search(query_text=concept_name, limit=context_chunk_limit, search_type="semantic")
        #     # Add them if they are different and valid
        #     # This logic needs careful handling of duplicates and relevance.
        #     # For now, we stick to the single chunk's context.
        
        context_chunks_for_generator = [{"chunk_text": context_for_qg_text}] # RAG QG expects list of dicts

        print(f"QuestionSelector: Generating {difficulty} question for concept '{concept_name}' (ID: {concept_id}).")
        
        generated_questions = await self.question_generator.generate_questions(
            context_chunks=context_chunks_for_generator,
            num_questions=1, 
            question_type="conceptual", # Or make this adaptive too
            difficulty_level=difficulty 
        )

        if not generated_questions:
            print(f"QuestionSelector: Failed to generate question for concept '{concept_name}'.")
            return None

        question_text = generated_questions[0]
        
        print(f"QuestionSelector: Selected question: \"{question_text}\"")
        return {
            "concept_id": concept_id, 
            "concept_name": concept_name,
            "question_text": question_text,
            "context_for_evaluation": context_for_qg_text 
        }

