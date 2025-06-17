# src/pipeline.py
import os
import asyncio
import json
import time 

from src.data_ingestion import document_loader
from src.data_ingestion import concept_tagger
from src.data_ingestion import chunker
from src.data_ingestion import vector_store_manager
from src.retrieval import retriever
from src.generation import question_generator_rag
from src.learner_model import profile_manager 
from src.evaluation import answer_evaluator 
from src.learner_model import knowledge_tracker 
from src.interaction import answer_handler 
from src.adaptive_engine import question_selector
from src.adaptive_engine.srs_scheduler import SRSScheduler 
from src import config
from typing import Optional

# Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEMO_LEARNER_ID = "learner_pipeline_interactive_001"
WEAVIATE_INDEXING_WAIT_TIME = 5 


async def run_ingestion_pipeline(processed_log_path: str):
    print("\n--- Phase 1: Data Ingestion & Storage (LaTeX Only) ---")
    client = None 
    try:
        client = vector_store_manager.get_weaviate_client()
        print("Weaviate client obtained for ingestion phase.")
        vector_store_manager.create_weaviate_schema(client)
        print("Weaviate schema ensured for ingestion phase.")
    except Exception as e:
        print(f"Could not connect to Weaviate or ensure schema: {e}")
        return None
    print("\nStep 1.1: Loading and Parsing LaTeX Documents...")
    latex_documents_path = config.DATA_DIR_RAW_LATEX
    if not os.path.isdir(latex_documents_path):
        print(f"ERROR: LaTeX documents directory not found at {latex_documents_path}.")
        return client 
    try:
        print(f"Attempting to load only LaTeX documents from {latex_documents_path} (skipping already processed)...")
        all_parsed_docs = document_loader.load_and_parse_documents(
            process_pdfs=False, processed_docs_log_path=processed_log_path
        )
        parsed_docs_data = [doc for doc in all_parsed_docs if doc and doc.get("original_type") == "latex"]
    except Exception as e:
        print(f"ERROR: Failed during LaTeX document loading/parsing phase: {e}")
        return client 
    if not parsed_docs_data:
        print("No new LaTeX documents to process for ingestion.")
        return client 
    print(f"Successfully parsed {len(parsed_docs_data)} new LaTeX documents.")
    print("\nStep 1.2: Concept/Topic Identification & Tagging...")
    all_conceptual_blocks = concept_tagger.tag_all_documents(parsed_docs_data)
    if not all_conceptual_blocks:
        if parsed_docs_data:
            print("ERROR: New LaTeX documents were parsed, but no conceptual blocks were identified. Skipping ingestion.")
            return client 
        print("Warning: No conceptual blocks identified.")
    else:
        print(f"Identified {len(all_conceptual_blocks)} conceptual blocks from new LaTeX documents.")
    if not all_conceptual_blocks: return client
    print("\nStep 1.3: Text Chunking...")
    final_text_chunks = chunker.chunk_conceptual_blocks(
        all_conceptual_blocks, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )
    if not final_text_chunks:
        print("ERROR: No text chunks were created. Skipping ingestion.")
        return client
    print(f"Created {len(final_text_chunks)} final text chunks from new LaTeX content.")
    print("\nStep 1.4 & 1.5: Embed and Store Chunks...")
    try:
        vector_store_manager.embed_and_store_chunks(client, final_text_chunks)
        print(f"Data ingestion complete for new LaTeX content.")
        newly_ingested_filenames = list(set([doc['filename'] for doc in parsed_docs_data if 'filename' in doc]))
        document_loader.update_processed_docs_log(processed_log_path, newly_ingested_filenames)
    except Exception as e:
        print(f"ERROR: Error during Weaviate data storage: {e}")
    return client


async def run_interaction_pipeline(
    client, 
    learner_id: str, 
    interactive_mode: bool = False,
    target_topic_id: Optional[str] = None 
    ):
    print("\n\n--- Phase 2, 3 & 5: Retrieval, Question Selection, Generation & Learner Interaction ---")
    if not client:
        print("Weaviate client not available. Cannot proceed.")
        return

    pm = None 
    try:
        print("\nInitializing components for interaction phase...")
        doc_retriever = retriever.Retriever(weaviate_client=client)
        q_generator = question_generator_rag.RAGQuestionGenerator()
        profile_db_path = os.path.join(config.DATA_DIR, f"{learner_id}_profile.sqlite3")
        pm = profile_manager.LearnerProfileManager(db_path=profile_db_path)
        
        srs_scheduler_instance = SRSScheduler()
        q_selector = question_selector.QuestionSelector(
            profile_manager=pm,
            retriever=doc_retriever,
            question_generator=q_generator
        )
        ans_evaluator = answer_evaluator.AnswerEvaluator()
        knowledge_track = knowledge_tracker.KnowledgeTracker(
            profile_manager=pm, 
            srs_scheduler=srs_scheduler_instance
        )
        ans_handler = answer_handler.AnswerHandler(evaluator=ans_evaluator, tracker=knowledge_track)
        print("Interaction components (including SRS Scheduler) initialized.")

        for interaction_count in range(1): 
            print(f"\n--- Interaction Cycle {interaction_count + 1} ---")
            print(f"Step 2.X: Selecting next question for learner '{learner_id}'" + (f" within topic '{target_topic_id}'." if target_topic_id else "."))
            
            next_question_info = await q_selector.select_next_question(learner_id, target_doc_id=target_topic_id)

            if not next_question_info or "error" in next_question_info:
                error_msg = next_question_info.get("error", "QuestionSelector could not select a next question.") if next_question_info else "QuestionSelector returned None."
                suggestion = next_question_info.get("suggestion", "") if next_question_info else ""
                print(f"QuestionSelector Info: {error_msg} {suggestion}")
                break 

            current_question_text = next_question_info["question_text"]
            question_concept_id = next_question_info["concept_id"] 
            context_for_evaluation = next_question_info["context_for_evaluation"]
            concept_name = next_question_info["concept_name"]
            # --- Get doc_id from question_info ---
            doc_id_for_tracking = next_question_info.get("doc_id", "unknown_doc_id_from_qselector") # Fallback
            if doc_id_for_tracking == "unknown_doc_id_from_qselector":
                # Attempt to find it in curriculum map if QuestionSelector didn't directly return it
                # (though it should based on question_selector_v2_curriculum)
                concept_block_details = next((block for block in q_selector.curriculum_map if block["concept_id"] == question_concept_id), None)
                if concept_block_details:
                    doc_id_for_tracking = concept_block_details.get("doc_id", "unknown_doc_id_after_lookup")

            print("\n--- Learner Interaction ---")
            print(f"Learner ID: {learner_id}")
            print(f"Selected Concept: {concept_name} (ID: {question_concept_id}, Doc: {doc_id_for_tracking})")
            print(f"Presenting Question:\n  Q: {current_question_text}")
            
            learner_actual_answer = None
            if interactive_mode:
                learner_actual_answer = input("Your Answer: ").strip()
                if not learner_actual_answer: 
                    print("No answer provided. Skipping evaluation for this question.")
                    continue 
            else:
                learner_actual_answer = "A vector space is a set of vectors that can be added together and multiplied by scalars, following certain axioms like closure under addition and scalar multiplication."
                print(f"Using Simulated Answer (non-interactive mode): \"{learner_actual_answer}\"")

            if learner_actual_answer: 
                handler_response = await ans_handler.submit_answer(
                    learner_id=learner_id, question_id=question_concept_id, 
                    doc_id=doc_id_for_tracking, # <<< PASS doc_id
                    question_text=current_question_text, retrieved_context=context_for_evaluation, 
                    learner_answer=learner_actual_answer
                )
                print("\n--- Evaluation & Tracking Results ---")
                print(f"  Feedback: {handler_response.get('feedback')}")
                print(f"  Accuracy: {handler_response.get('accuracy_score')}")
                print(f"  Suggested Correct Answer: {handler_response.get('correct_answer_suggestion')}")
                
                updated_knowledge = pm.get_concept_knowledge(learner_id, question_concept_id)
                if updated_knowledge: 
                    print(f"  Updated Knowledge for '{concept_name}':")
                    print(f"    Current Score (0-10): {updated_knowledge.get('current_score')}")
                    print(f"    Total Attempts: {updated_knowledge.get('total_attempts')}")
                    print(f"    SRS Reps: {updated_knowledge.get('srs_repetitions')}")
                    print(f"    SRS Interval (days): {updated_knowledge.get('srs_interval_days')}")
                    print(f"    Next Review At: {updated_knowledge.get('next_review_at')}")
                    print(f"    Current Difficulty: {updated_knowledge.get('current_difficulty_level')}")
            
            if not interactive_mode: break

    except Exception as e:
        print(f"Error during interaction pipeline: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pm:
            pm.close_db()


async def run_full_pipeline(interactive_mode: bool = False, 
                            initial_learner_id: Optional[str] = None,
                            target_topic_id: Optional[str] = None): 
    print("Starting RAG System - Full Pipeline Execution...")
    processed_log_path = config.PROCESSED_DOCS_LOG_FILE
    weaviate_client = await run_ingestion_pipeline(processed_log_path)

    if weaviate_client:
        print(f"\nWaiting {WEAVIATE_INDEXING_WAIT_TIME} seconds for Weaviate to index...")
        await asyncio.sleep(WEAVIATE_INDEXING_WAIT_TIME)
        learner_id_to_use = initial_learner_id if initial_learner_id else DEMO_LEARNER_ID
        await run_interaction_pipeline(
            client=weaviate_client, 
            learner_id=learner_id_to_use,
            interactive_mode=interactive_mode,
            target_topic_id=target_topic_id 
            )
    else:
        print("Ingestion phase failed or Weaviate client not available. Skipping interaction phase.")

    print("\n\n--- RAG System Pipeline Finished ---")
    await asyncio.sleep(0.25)

if __name__ == '__main__':
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(dotenv_path):
        from dotenv import load_dotenv
        print(f"pipeline.py: Found .env file at {dotenv_path}, loading.")
        load_dotenv(dotenv_path)
    asyncio.run(run_full_pipeline(interactive_mode=False)) 
