# src/app.py
import os
import asyncio
from src import pipeline # Import the new pipeline module
from src import config # To ensure config is loaded, used by pipeline

def setup_environment():
    """Loads environment variables from .env file if it exists."""
    # .env path relative to this app.py file (src/.env or project_root/.env)
    # Assuming .env is in the project root, one level up from src/
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(dotenv_path):
        print(f"Found .env file at {dotenv_path}, attempting to load environment variables.")
        from dotenv import load_dotenv # Local import
        load_dotenv(dotenv_path)
        print("Environment variables from .env potentially loaded.")
    else:
        print(f".env file not found at {dotenv_path}. Relying on system environment variables.")

if __name__ == "__main__":
    setup_environment() # Load .env before pipeline uses config

    try:
        asyncio.run(pipeline.run_full_pipeline())
    except Exception as e:
        print(f"An error occurred in the main application flow: {e}")
        import traceback
        traceback.print_exc()
