# src/pipeline.py
import os
import asyncio
import json

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
from src import config

# Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_QUERY_FOR_CONTEXT = "What is a vector space?"
DEFAULT_NUM_RETRIEVED_CHUNKS_FOR_QUESTION_CONTEXT = 1
DEFAULT_NUM_QUESTIONS_TO_GENERATE = 1
DEFAULT_QUESTION_TYPE = "conceptual"
DEFAULT_SEARCH_TYPE = "hybrid"
DEMO_LEARNER_ID = "learner_pipeline_interactive_001"


async def run_ingestion_pipeline(processed_log_path: str):
    """
    Runs the data ingestion part of the pipeline (Phase 1).
    Processes new LaTeX documents and stores them in Weaviate.
    Returns the Weaviate client instance if successful, or None.
    """
    print("\n--- Phase 1: Data Ingestion & Storage (LaTeX Only) ---")
    print("\nStep 1.1: Loading and Parsing LaTeX Documents...")
    
    latex_documents_path = config.DATA_DIR_RAW_LATEX
    if not os.path.isdir(latex_documents_path):
        print(f"ERROR: LaTeX documents directory not found at {latex_documents_path}.")
        return None

    try:
        print(f"Attempting to load only LaTeX documents from {latex_documents_path} (skipping already processed)...")
        all_parsed_docs = document_loader.load_and_parse_documents(
            process_pdfs=False,
            processed_docs_log_path=processed_log_path
        )
        parsed_docs_data = [doc for doc in all_parsed_docs if doc and doc.get("original_type") == "latex"]
    except Exception as e:
        print(f"ERROR: Failed during LaTeX document loading/parsing phase: {e}")
        import traceback
        traceback.print_exc()
        return None

    if not parsed_docs_data:
        print("No new LaTeX documents to process for ingestion.")
        try:
            client = vector_store_manager.get_weaviate_client()
            print("Weaviate client obtained. No new documents were ingested in this run.")
            return client
        except Exception as e:
            print(f"Could not connect to Weaviate: {e}")
            return None
            
    print(f"Successfully parsed {len(parsed_docs_data)} new LaTeX documents.")

    print("\nStep 1.2: Concept/Topic Identification & Tagging...")
    all_conceptual_blocks = concept_tagger.tag_all_documents(parsed_docs_data)
    if not all_conceptual_blocks:
        if parsed_docs_data:
            print("ERROR: New LaTeX documents were parsed, but no conceptual blocks were identified. Exiting ingestion.")
            return None
        print("Warning: No conceptual blocks identified.")
    else:
        print(f"Identified {len(all_conceptual_blocks)} conceptual blocks from new LaTeX documents.")

    if not all_conceptual_blocks: 
        try: 
            client = vector_store_manager.get_weaviate_client()
            return client
        except Exception as e:
            print(f"Could not connect to Weaviate: {e}")
            return None

    print("\nStep 1.3: Text Chunking...")
    final_text_chunks = chunker.chunk_conceptual_blocks(
        all_conceptual_blocks,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )
    if not final_text_chunks:
        print("ERROR: No text chunks were created from the new LaTeX conceptual blocks. Exiting ingestion.")
        return None
    print(f"Created {len(final_text_chunks)} final text chunks from new LaTeX content.")

    print("\nStep 1.4 & 1.5: Initialize Weaviate, Embed and Store Chunks...")
    try:
        client = vector_store_manager.get_weaviate_client()
        vector_store_manager.embed_and_store_chunks(client, final_text_chunks)
        print(f"Data ingestion complete for new LaTeX content. Chunks stored in Weaviate class: {vector_store_manager.WEAVIATE_CLASS_NAME}")
        
        newly_ingested_filenames = list(set([doc['filename'] for doc in parsed_docs_data if 'filename' in doc]))
        document_loader.update_processed_docs_log(processed_log_path, newly_ingested_filenames)
        return client
    except ConnectionError as e:
        print(f"ERROR: Could not connect to Weaviate: {e}. Please ensure Weaviate is running.")
        return None
    except Exception as e:
        print(f"ERROR: Error during Weaviate initialization or data storage: {e}")
        import traceback
        traceback.print_exc()
        return None


async def run_interaction_pipeline(
    client, 
    learner_id: str, 
    query_for_context: str,
    interactive_mode: bool = False # New flag to control input
    ):
    """
    Runs retrieval, question generation, and optionally interactive answer submission.
    """
    print("\n\n--- Phase 2 & 3: Retrieval, Question Generation & Learner Interaction ---")
    if not client:
        print("Weaviate client not available. Cannot proceed.")
        return

    pm = None # Initialize pm to None for finally block
    try:
        print("\nInitializing components for interaction phase...")
        doc_retriever = retriever.Retriever(
            weaviate_client=client,
            weaviate_class_name=vector_store_manager.WEAVIATE_CLASS_NAME
        )
        q_generator = question_generator_rag.RAGQuestionGenerator()
        
        profile_db_path = os.path.join(config.DATA_DIR, f"{learner_id}_profile.sqlite3")
        pm = profile_manager.LearnerProfileManager(db_path=profile_db_path)
        pm.create_profile(learner_id)

        ans_evaluator = answer_evaluator.AnswerEvaluator()
        knowledge_track = knowledge_tracker.KnowledgeTracker(profile_manager=pm)
        ans_handler = answer_handler.AnswerHandler(evaluator=ans_evaluator, tracker=knowledge_track)
        print("Interaction components initialized.")

        print(f"\nStep 2.2: Retrieving context for query: '{query_for_context}'...")
        retrieved_chunks_for_qg = doc_retriever.search(
            query_text=query_for_context,
            search_type=DEFAULT_SEARCH_TYPE,
            limit=DEFAULT_NUM_RETRIEVED_CHUNKS_FOR_QUESTION_CONTEXT,
            additional_properties=["id", "score", "certainty"]
        )

        if not retrieved_chunks_for_qg:
            print("No relevant chunks retrieved for question generation context.")
            return
        
        print(f"Successfully retrieved {len(retrieved_chunks_for_qg)} chunk(s) for question generation context.")
        context_for_qg = "\n\n".join([chunk.get("chunk_text", "") for chunk in retrieved_chunks_for_qg])

        print(f"\nStep 2.4: Generating {DEFAULT_NUM_QUESTIONS_TO_GENERATE} '{DEFAULT_QUESTION_TYPE}' question(s)...")
        generated_questions = await q_generator.generate_questions(
            context_chunks=retrieved_chunks_for_qg,
            num_questions=DEFAULT_NUM_QUESTIONS_TO_GENERATE,
            question_type=DEFAULT_QUESTION_TYPE
        )

        if not generated_questions:
            print("No questions were generated. Ending interaction phase.")
            return

        current_question_text = generated_questions[0]
        question_concept_id = retrieved_chunks_for_qg[0].get("_id", 
                                    retrieved_chunks_for_qg[0].get("chunk_id", 
                                                                    "unknown_concept_from_pipeline")) 
        
        print("\n--- Learner Interaction ---")
        print(f"Learner ID: {learner_id}")
        print(f"Presenting Question (Concept ID: {question_concept_id}):\n  Q: {current_question_text}")
        
        learner_actual_answer = None
        if interactive_mode:
            learner_actual_answer = input("Your Answer: ").strip()
        else:
            # Fallback to a simulated answer if not interactive for this demo part
            learner_actual_answer = "A vector space is a set of vectors that can be added together and multiplied by scalars, following certain axioms like closure under addition and scalar multiplication." # SIMULATED_LEARNER_ANSWER_GOOD
            print(f"Using Simulated Answer (non-interactive mode): \"{learner_actual_answer}\"")


        if learner_actual_answer:
            handler_response = await ans_handler.submit_answer(
                learner_id=learner_id,
                question_id=question_concept_id,
                question_text=current_question_text,
                retrieved_context=context_for_qg,
                learner_answer=learner_actual_answer
            )
            print("\n--- Evaluation & Tracking Results ---")
            print(f"  Feedback from Evaluator: {handler_response.get('feedback')}")
            print(f"  Accuracy Score (0-1): {handler_response.get('accuracy_score')}")
            
            updated_knowledge = pm.get_concept_knowledge(learner_id, question_concept_id)
            if updated_knowledge:
                print(f"  Updated Knowledge for '{question_concept_id}':")
                print(f"    Current Score (0-10): {updated_knowledge.get('current_score')}")
                print(f"    Total Attempts: {updated_knowledge.get('total_attempts')}")
                print(f"    Correct Attempts: {updated_knowledge.get('correct_attempts')}")
                print(f"    Last Answered Correctly: {'Yes' if updated_knowledge.get('last_answered_correctly') else 'No'}")
            else:
                print(f"  Could not retrieve updated knowledge for concept '{question_concept_id}'.")
        else:
            print("No answer provided by learner, skipping submission and evaluation.")

    except Exception as e:
        print(f"Error during interaction pipeline: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pm:
            pm.close_db()


async def run_full_pipeline(interactive_mode: bool = False):
    """
    Runs the complete RAG pipeline.
    """
    print("Starting RAG System - Full Pipeline Execution...")
    
    processed_log_path = config.PROCESSED_DOCS_LOG_FILE
    
    weaviate_client = await run_ingestion_pipeline(processed_log_path)

    if weaviate_client:
        await run_interaction_pipeline(
            client=weaviate_client, 
            learner_id=DEMO_LEARNER_ID, # Could also be prompted for in app.py
            query_for_context=DEFAULT_QUERY_FOR_CONTEXT,
            interactive_mode=interactive_mode # Pass the flag
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
    
    # Example: Run in non-interactive mode by default if run directly
    # Change to True to test interactive input() from pipeline.py
    asyncio.run(run_full_pipeline(interactive_mode=False)) 
