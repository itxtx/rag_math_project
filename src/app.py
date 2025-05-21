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
    Main function to run the RAG pipeline with interactive input for answers.
    """
    print("Starting RAG System - Interactive Application Mode...")
    setup_environment() 

    # --- Interactive Learner Session ---
    print("\n\n--- Interactive Learner Session ---")
    
    default_learner_id = pipeline.DEMO_LEARNER_ID # Get default from pipeline module
    learner_id_input = input(f"Enter your Learner ID (default: '{default_learner_id}'): ").strip()
    learner_id = learner_id_input if learner_id_input else default_learner_id
    print(f"Using Learner ID: {learner_id}")
    
    # The QuestionSelector will now determine the topic and question.
    # The 'query_for_context' input is no longer directly needed here.
    
    # Run the full pipeline, which now includes adaptive question selection
    # and will prompt for an answer if interactive_mode is True.
    await pipeline.run_full_pipeline(
        interactive_mode=True,
        initial_learner_id=learner_id
    )
    
    print("\n--- End of Interactive Session ---")
    # The sleep for resource cleanup is now in pipeline.run_full_pipeline


if __name__ == "__main__":
    try:
        asyncio.run(main_interactive_app())
    except KeyboardInterrupt:
        print("\nApplication interrupted by user. Exiting.")
    except Exception as e:
        print(f"An error occurred in the main application flow: {e}")
        import traceback
        traceback.print_exc()
