#  src/app.py
import os
import asyncio
import sys 
from src import pipeline 
from src import config 
# For API call to list topics, if we decide to fetch from API
# import httpx 

def setup_environment():
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(dotenv_path):
        print(f"Found .env file at {dotenv_path}, attempting to load environment variables.")
        from dotenv import load_dotenv 
        load_dotenv(dotenv_path)
        print("Environment variables from .env potentially loaded.")
    else:
        print(f".env file not found at {dotenv_path}. Relying on system environment variables.")

async def main_interactive_app():
    print("Starting RAG System - Interactive Application Mode...")
    setup_environment() 

    # --- Initialize components needed to list topics ---
    # This is a bit of a workaround for a CLI app. A real UI would call the API.
    # For this CLI demo, we'll instantiate components to get topics.
    temp_pm = None
    temp_retriever = None
    q_selector_for_topics = None
    weaviate_client_for_topics = None
    available_topics_list = []

    try:
        weaviate_client_for_topics = pipeline.vector_store_manager.get_weaviate_client()
        pipeline.vector_store_manager.create_weaviate_schema(weaviate_client_for_topics) # Ensure schema
        
        temp_pm = pipeline.profile_manager.LearnerProfileManager() # Temp instance for topic listing
        temp_retriever = pipeline.retriever.Retriever(weaviate_client=weaviate_client_for_topics)
        q_selector_for_topics = pipeline.question_selector.QuestionSelector(
            profile_manager=temp_pm, # Not strictly needed for get_available_topics if map loads from retriever
            retriever=temp_retriever,
            question_generator=pipeline.question_generator_rag.RAGQuestionGenerator() # Placeholder QG
        )
        available_topics_list = q_selector_for_topics.get_available_topics()
    except Exception as e:
        print(f"Warning: Could not fetch available topics: {e}. Proceeding without topic selection.")
    finally:
        if temp_pm: temp_pm.close_db()
        # We don't close weaviate_client_for_topics here as the main pipeline might use it.
        # The main pipeline will create its own client instance.

    # --- Interactive Learner Session ---
    print("\n\n--- Interactive Learner Session ---")
    
    default_learner_id = pipeline.DEMO_LEARNER_ID
    learner_id_input = input(f"Enter your Learner ID (default: '{default_learner_id}'): ").strip()
    learner_id = learner_id_input if learner_id_input else default_learner_id
    print(f"Using Learner ID: {learner_id}")
    
    target_topic_id_input = None
    if available_topics_list:
        print("\nAvailable Topics:")
        for i, topic_info in enumerate(available_topics_list):
            print(f"  {i+1}. {topic_info.get('topic_id')} (Source: {topic_info.get('source_file')})")
        
        while True:
            try:
                choice = input(f"Choose a topic number (or press Enter to let system choose adaptively): ").strip()
                if not choice: # User pressed Enter
                    target_topic_id_input = None
                    break
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_topics_list):
                    target_topic_id_input = available_topics_list[choice_idx]['topic_id']
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number or press Enter.")
        print(f"Selected topic for questions: {target_topic_id_input if target_topic_id_input else 'Adaptive (any topic)'}")
    else:
        print("No topics loaded, proceeding with adaptive selection across all content (if any).")

    await pipeline.run_full_pipeline(
        interactive_mode=True,
        initial_learner_id=learner_id,
        target_topic_id=target_topic_id_input # Pass the chosen topic
    )
    
    print("\n--- End of Interactive Session ---")

if __name__ == "__main__":
    try:
        asyncio.run(main_interactive_app())
    except KeyboardInterrupt:
        print("\nApplication interrupted by user. Exiting.")
    except Exception as e:
        print(f"An error occurred in the main application flow: {e}")
        import traceback
        traceback.print_exc()
