# src/app.py
import os
import asyncio
import sys 
from src import pipeline 
from src import config 

def setup_environment():
    """Loads environment variables from .env file if it exists."""
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(dotenv_path):
        print(f"Found .env file at {dotenv_path}, attempting to load environment variables.")
        from dotenv import load_dotenv 
        load_dotenv(dotenv_path)
        print("Environment variables from .env potentially loaded.")
    else:
        print(f".env file not found at {dotenv_path}. Relying on system environment variables.")

async def main_interactive_app():
    """
    Main function to run the RAG pipeline, prompting for learner ID and context query,
    and then running the interaction pipeline in interactive mode.
    """
    print("Starting RAG System - Interactive Application Mode...")
    setup_environment() 

    processed_log_path = config.PROCESSED_DOCS_LOG_FILE
    
    # Run Ingestion Phase first
    weaviate_client = await pipeline.run_ingestion_pipeline(processed_log_path)

    if not weaviate_client:
        print("Ingestion phase failed or Weaviate client not available. Cannot proceed to interaction.")
        return

    # --- Interactive Learner Session ---
    print("\n\n--- Interactive Learner Session ---")
    
    default_learner_id = pipeline.DEMO_LEARNER_ID
    learner_id_input = input(f"Enter your Learner ID (default: '{default_learner_id}'): ").strip()
    learner_id = learner_id_input if learner_id_input else default_learner_id
    print(f"Using Learner ID: {learner_id}")

    default_context_query = pipeline.DEFAULT_QUERY_FOR_CONTEXT
    query_for_context_input = input(f"Enter a topic/query to find context for a question (default: '{default_context_query}'): ").strip()
    query_for_context = query_for_context_input if query_for_context_input else default_context_query
    print(f"Using query for context: '{query_for_context}'")
    
    # Now call the interaction pipeline in interactive mode
    await pipeline.run_interaction_pipeline(
        client=weaviate_client,
        learner_id=learner_id,
        query_for_context=query_for_context,
        interactive_mode=True # Explicitly set to True
    )
    
    print("\n--- End of Interactive Session ---")
    await asyncio.sleep(0.25)


if __name__ == "__main__":
    try:
        asyncio.run(main_interactive_app()) # Call the new interactive main function
    except KeyboardInterrupt:
        print("\nApplication interrupted by user. Exiting.")
    except Exception as e:
        print(f"An error occurred in the main application flow: {e}")
        import traceback
        traceback.print_exc()
