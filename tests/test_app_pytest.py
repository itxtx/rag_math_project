import pytest
from unittest.mock import patch, mock_open, MagicMock, AsyncMock
from io import StringIO
import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.app import setup_environment, main_interactive_app

# --- Fixtures ---

@pytest.fixture
def mock_env_file():
    return "API_KEY=test_api_key_123\nDATABASE_URL=test_db_url\n"

@pytest.fixture
def mock_input_success():
    return StringIO("test_learner\n1\n")

@pytest.fixture
def mock_input_no_topic():
    return StringIO("test_learner\n\n")

@pytest.fixture
def mock_input_invalid_topic():
    return StringIO("test_learner\n99\n1\n")

@pytest.fixture
def mock_input_invalid_learner():
    return StringIO("\n123\n")

# --- Environment Setup Tests ---

def test_setup_environment_with_env_file(mock_env_file):
    """Test environment setup when .env file exists."""
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_env_file)), \
         patch('dotenv.load_dotenv') as mock_load_dotenv:
        
        setup_environment()
        
        mock_load_dotenv.assert_called_once()

def test_setup_environment_without_env_file():
    """Test environment setup when .env file doesn't exist."""
    with patch('os.path.exists', return_value=False), \
         patch('builtins.print') as mock_print:
        
        setup_environment()
        
        assert any(".env file not found at" in call.args[0] for call in mock_print.call_args_list)

def test_setup_environment_env_file_path():
    """Test that the correct .env file path is used."""
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data="")), \
         patch('dotenv.load_dotenv') as mock_load_dotenv:
        
        setup_environment()
        
        mock_load_dotenv.assert_called_once()

# --- Interactive Application Tests ---

@pytest.mark.asyncio
async def test_main_interactive_app_success(mock_input_success):
    """Test successful interactive application flow."""
    with patch('sys.stdin', mock_input_success), \
         patch('builtins.input', side_effect=mock_input_success.getvalue().split('\n')[:-1]), \
         patch('src.app.pipeline') as mock_pipeline:
        
        mock_pipeline.vector_store_manager.get_weaviate_client.return_value = MagicMock()
        mock_pipeline.vector_store_manager.create_weaviate_schema.return_value = None
        mock_pipeline.profile_manager.LearnerProfileManager.return_value = MagicMock()
        
        mock_retriever = AsyncMock()
        mock_pipeline.retriever.HybridRetriever.return_value = mock_retriever
        
        mock_question_selector = AsyncMock()
        mock_question_selector.curriculum_map = [{"doc_id": "topic1", "concept_name": "Topic 1"}]
        mock_pipeline.question_selector.QuestionSelector.return_value = mock_question_selector

        mock_pipeline.run_full_pipeline = AsyncMock()

        await main_interactive_app()

        mock_pipeline.run_full_pipeline.assert_awaited_once_with(
            interactive_mode=True, 
            initial_learner_id="test_learner", 
            target_topic_id="topic1"
        )

@pytest.mark.asyncio
async def test_main_interactive_app_no_topics_available(mock_input_no_topic):
    """Test interactive application when no topics are available."""
    with patch('sys.stdin', mock_input_no_topic), \
         patch('builtins.input', side_effect=mock_input_no_topic.getvalue().split('\n')[:-1]), \
         patch('src.app.pipeline') as mock_pipeline, \
         patch('builtins.print') as mock_print:
        
        # Mock pipeline components
        mock_pipeline.vector_store_manager.get_weaviate_client.return_value = MagicMock()
        mock_pipeline.vector_store_manager.create_weaviate_schema.return_value = None
        mock_pipeline.profile_manager.LearnerProfileManager.return_value = MagicMock()
        mock_pipeline.retriever.Retriever.return_value = MagicMock()
        mock_pipeline.question_generator_rag.RAGQuestionGenerator.return_value = MagicMock()
        
        # Mock question selector with no topics
        mock_question_selector = MagicMock()
        mock_question_selector.get_available_topics.return_value = []
        mock_pipeline.question_selector.QuestionSelector.return_value = mock_question_selector
        
        # Mock the main pipeline run
        mock_pipeline.run_full_pipeline = AsyncMock()
        
        await main_interactive_app()
        
        # Verify that the pipeline was called with no target topic
        mock_pipeline.run_full_pipeline.assert_awaited_once_with(
            interactive_mode=True,
            initial_learner_id="test_learner",
            target_topic_id=None
        )

@pytest.mark.asyncio
async def test_main_interactive_app_topic_selection_error(mock_input_success):
    """Test interactive application when topic selection fails."""
    with patch('sys.stdin', mock_input_success), \
         patch('builtins.input', side_effect=mock_input_success.getvalue().split('\n')[:-1]), \
         patch('src.app.pipeline') as mock_pipeline, \
         patch('builtins.print') as mock_print:
        
        # Mock pipeline components with error
        mock_pipeline.vector_store_manager.get_weaviate_client.side_effect = Exception("Connection failed")
        
        # Mock the main pipeline run
        mock_pipeline.run_full_pipeline = AsyncMock()
        
        await main_interactive_app()
        
        # Verify that the pipeline was called even when topic selection fails
        mock_pipeline.run_full_pipeline.assert_awaited_once_with(
            interactive_mode=True,
            initial_learner_id="test_learner",
            target_topic_id=None
        )

@pytest.mark.asyncio
async def test_main_interactive_app_invalid_topic_choice(mock_input_invalid_topic):
    """Test interactive application with invalid topic choice."""
    with patch('sys.stdin', mock_input_invalid_topic), \
         patch('builtins.input', side_effect=mock_input_invalid_topic.getvalue().split('\n')[:-1]), \
         patch('src.app.pipeline') as mock_pipeline, \
         patch('builtins.print') as mock_print:
        
        # Mock pipeline components
        mock_pipeline.vector_store_manager.get_weaviate_client.return_value = MagicMock()
        mock_pipeline.vector_store_manager.create_weaviate_schema.return_value = None
        mock_pipeline.profile_manager.LearnerProfileManager.return_value = MagicMock()
        mock_pipeline.retriever.Retriever.return_value = MagicMock()
        mock_pipeline.question_generator_rag.RAGQuestionGenerator.return_value = MagicMock()
        
        # Mock question selector
        mock_question_selector = MagicMock()
        mock_question_selector.get_available_topics.return_value = [
            {"topic_id": "topic1", "source_file": "file1.pdf"}
        ]
        mock_pipeline.question_selector.QuestionSelector.return_value = mock_question_selector
        
        # Mock the main pipeline run
        mock_pipeline.run_full_pipeline = AsyncMock()
        
        await main_interactive_app()
        
        # Verify that the pipeline was called with the valid topic choice
        mock_pipeline.run_full_pipeline.assert_awaited_once_with(
            interactive_mode=True,
            initial_learner_id="test_learner",
            target_topic_id="topic1"
        )

@pytest.mark.asyncio
async def test_main_interactive_app_default_learner_id():
    """Test interactive application with default learner ID."""
    with patch('builtins.input', return_value=""), \
         patch('src.app.pipeline') as mock_pipeline, \
         patch('builtins.print') as mock_print:
        
        # Mock pipeline components
        mock_pipeline.vector_store_manager.get_weaviate_client.return_value = MagicMock()
        mock_pipeline.vector_store_manager.create_weaviate_schema.return_value = None
        mock_pipeline.profile_manager.LearnerProfileManager.return_value = MagicMock()
        mock_pipeline.retriever.Retriever.return_value = MagicMock()
        mock_pipeline.question_generator_rag.RAGQuestionGenerator.return_value = MagicMock()
        
        # Mock question selector with no topics
        mock_question_selector = MagicMock()
        mock_question_selector.get_available_topics.return_value = []
        mock_pipeline.question_selector.QuestionSelector.return_value = mock_question_selector
        
        # Mock the main pipeline run
        mock_pipeline.run_full_pipeline = AsyncMock()
        
        await main_interactive_app()
        
        # Verify that the pipeline was called with default learner ID
        mock_pipeline.run_full_pipeline.assert_awaited_once_with(
            interactive_mode=True,
            initial_learner_id=mock_pipeline.DEMO_LEARNER_ID,
            target_topic_id=None
        )

@pytest.mark.asyncio
async def test_main_interactive_app_pipeline_error():
    """Test interactive application when pipeline raises an exception."""
    with patch('builtins.input', return_value="test_learner"), \
         patch('src.app.pipeline') as mock_pipeline, \
         patch('builtins.print') as mock_print:
        
        # Mock pipeline components
        mock_pipeline.vector_store_manager.get_weaviate_client.return_value = MagicMock()
        mock_pipeline.vector_store_manager.create_weaviate_schema.return_value = None
        mock_pipeline.profile_manager.LearnerProfileManager.return_value = MagicMock()
        mock_pipeline.retriever.Retriever.return_value = MagicMock()
        mock_pipeline.question_generator_rag.RAGQuestionGenerator.return_value = MagicMock()
        
        # Mock question selector with no topics
        mock_question_selector = MagicMock()
        mock_question_selector.get_available_topics.return_value = []
        mock_pipeline.question_selector.QuestionSelector.return_value = mock_question_selector
        
        # Mock the main pipeline run to raise an exception
        mock_pipeline.run_full_pipeline = AsyncMock(side_effect=Exception("Pipeline error"))
        
        with pytest.raises(Exception, match="Pipeline error"):
            await main_interactive_app()

# --- Resource Cleanup Tests ---

@pytest.mark.asyncio
async def test_main_interactive_app_profile_manager_cleanup():
    """Test that profile manager is properly cleaned up."""
    with patch('builtins.input', return_value="test_learner"), \
         patch('src.app.pipeline') as mock_pipeline, \
         patch('builtins.print') as mock_print:
        
        # Mock pipeline components
        mock_pipeline.vector_store_manager.get_weaviate_client.return_value = MagicMock()
        mock_pipeline.vector_store_manager.create_weaviate_schema.return_value = None
        
        # Mock profile manager with close_db method
        mock_profile_manager = MagicMock()
        mock_profile_manager.close_db = MagicMock()
        mock_pipeline.profile_manager.LearnerProfileManager.return_value = mock_profile_manager
        
        mock_pipeline.retriever.Retriever.return_value = MagicMock()
        mock_pipeline.question_generator_rag.RAGQuestionGenerator.return_value = MagicMock()
        
        # Mock question selector with no topics
        mock_question_selector = MagicMock()
        mock_question_selector.get_available_topics.return_value = []
        mock_pipeline.question_selector.QuestionSelector.return_value = mock_question_selector
        
        # Mock the main pipeline run
        mock_pipeline.run_full_pipeline = AsyncMock()
        
        await main_interactive_app()
        
        # Verify that profile manager was closed
        mock_profile_manager.close_db.assert_called_once()

@pytest.mark.asyncio
async def test_main_interactive_app_profile_manager_cleanup_on_error():
    """Test that profile manager is cleaned up even when topic selection fails."""
    with patch('builtins.input', return_value="test_learner"), \
         patch('src.app.pipeline') as mock_pipeline, \
         patch('builtins.print') as mock_print:
        
        # Mock pipeline components with error
        mock_pipeline.vector_store_manager.get_weaviate_client.side_effect = Exception("Connection failed")
        
        # Mock profile manager with close_db method
        mock_profile_manager = MagicMock()
        mock_profile_manager.close_db = MagicMock()
        mock_pipeline.profile_manager.LearnerProfileManager.return_value = mock_profile_manager
        
        # Mock the main pipeline run
        mock_pipeline.run_full_pipeline = AsyncMock()
        
        await main_interactive_app()
        
        # Verify that profile manager was closed even when error occurred
        mock_profile_manager.close_db.assert_called_once()

# --- Input Validation Tests ---

@pytest.mark.asyncio
async def test_main_interactive_app_invalid_learner_id_input(mock_input_invalid_learner):
    """Test interactive application with invalid learner ID input."""
    with patch('sys.stdin', mock_input_invalid_learner), \
         patch('builtins.input', side_effect=mock_input_invalid_learner.getvalue().split('\n')[:-1]), \
         patch('src.app.pipeline') as mock_pipeline, \
         patch('builtins.print') as mock_print:
        
        # Mock pipeline components
        mock_pipeline.vector_store_manager.get_weaviate_client.return_value = MagicMock()
        mock_pipeline.vector_store_manager.create_weaviate_schema.return_value = None
        mock_pipeline.profile_manager.LearnerProfileManager.return_value = MagicMock()
        mock_pipeline.retriever.Retriever.return_value = MagicMock()
        mock_pipeline.question_generator_rag.RAGQuestionGenerator.return_value = MagicMock()
        
        # Mock question selector with no topics
        mock_question_selector = MagicMock()
        mock_question_selector.get_available_topics.return_value = []
        mock_pipeline.question_selector.QuestionSelector.return_value = mock_question_selector
        
        # Mock the main pipeline run
        mock_pipeline.run_full_pipeline = AsyncMock()
        
        await main_interactive_app()
        
        # Verify that the pipeline was called with the valid learner ID
        mock_pipeline.run_full_pipeline.assert_awaited_once_with(
            interactive_mode=True,
            initial_learner_id="test_learner",
            target_topic_id=None
        )

# --- Edge Cases ---

@pytest.mark.asyncio
async def test_main_interactive_app_very_long_learner_id():
    """Test interactive application with very long learner ID."""
    long_learner_id = "x" * 1000
    
    with patch('builtins.input', return_value=long_learner_id), \
         patch('src.app.pipeline') as mock_pipeline, \
         patch('builtins.print') as mock_print:
        
        # Mock pipeline components
        mock_pipeline.vector_store_manager.get_weaviate_client.return_value = MagicMock()
        mock_pipeline.vector_store_manager.create_weaviate_schema.return_value = None
        mock_pipeline.profile_manager.LearnerProfileManager.return_value = MagicMock()
        mock_pipeline.retriever.Retriever.return_value = MagicMock()
        mock_pipeline.question_generator_rag.RAGQuestionGenerator.return_value = MagicMock()
        
        # Mock question selector with no topics
        mock_question_selector = MagicMock()
        mock_question_selector.get_available_topics.return_value = []
        mock_pipeline.question_selector.QuestionSelector.return_value = mock_question_selector
        
        # Mock the main pipeline run
        mock_pipeline.run_full_pipeline = AsyncMock()
        
        await main_interactive_app()
        
        # Verify that the pipeline was called with the long learner ID
        mock_pipeline.run_full_pipeline.assert_awaited_once_with(
            interactive_mode=True,
            initial_learner_id=long_learner_id,
            target_topic_id=None
        )

@pytest.mark.asyncio
async def test_main_interactive_app_special_characters_in_learner_id():
    """Test interactive application with special characters in learner ID."""
    special_learner_id = "learner_123!@#$%^&*()"
    
    with patch('builtins.input', return_value=special_learner_id), \
         patch('src.app.pipeline') as mock_pipeline, \
         patch('builtins.print') as mock_print:
        
        # Mock pipeline components
        mock_pipeline.vector_store_manager.get_weaviate_client.return_value = MagicMock()
        mock_pipeline.vector_store_manager.create_weaviate_schema.return_value = None
        mock_pipeline.profile_manager.LearnerProfileManager.return_value = MagicMock()
        mock_pipeline.retriever.Retriever.return_value = MagicMock()
        mock_pipeline.question_generator_rag.RAGQuestionGenerator.return_value = MagicMock()
        
        # Mock question selector with no topics
        mock_question_selector = MagicMock()
        mock_question_selector.get_available_topics.return_value = []
        mock_pipeline.question_selector.QuestionSelector.return_value = mock_question_selector
        
        # Mock the main pipeline run
        mock_pipeline.run_full_pipeline = AsyncMock()
        
        await main_interactive_app()
        
        # Verify that the pipeline was called with the special learner ID
        mock_pipeline.run_full_pipeline.assert_awaited_once_with(
            interactive_mode=True,
            initial_learner_id=special_learner_id,
            target_topic_id=None
        )

# --- Main Entry Point Tests ---

def test_main_entry_point_success():
    """Test the main entry point with successful execution."""
    with patch('asyncio.run') as mock_run, \
         patch('src.app.main_interactive_app') as mock_main:
        
        # Import and run the main block
        with patch('builtins.__import__', side_effect=__import__):
            # This simulates running the file directly
            if __name__ == "__main__":
                asyncio.run(main_interactive_app())
        
        mock_run.assert_called_once_with(mock_main.return_value)

def test_main_entry_point_keyboard_interrupt():
    """Test the main entry point with keyboard interrupt."""
    with patch('asyncio.run', side_effect=KeyboardInterrupt), \
         patch('builtins.print') as mock_print:
        
        # Import and run the main block
        with patch('builtins.__import__', side_effect=__import__):
            # This simulates running the file directly
            if __name__ == "__main__":
                try:
                    asyncio.run(main_interactive_app())
                except KeyboardInterrupt:
                    print("\nApplication interrupted by user. Exiting.")
        
        mock_print.assert_called_with("\nApplication interrupted by user. Exiting.")

def test_main_entry_point_general_exception():
    """Test the main entry point with general exception."""
    with patch('asyncio.run', side_effect=Exception("Test error")), \
         patch('builtins.print') as mock_print, \
         patch('traceback.print_exc') as mock_traceback:
        
        # Import and run the main block
        with patch('builtins.__import__', side_effect=__import__):
            # This simulates running the file directly
            if __name__ == "__main__":
                try:
                    asyncio.run(main_interactive_app())
                except Exception as e:
                    print(f"An error occurred in the main application flow: {e}")
                    import traceback
                    traceback.print_exc()
        
        mock_print.assert_called_with("An error occurred in the main application flow: Test error")
        mock_traceback.assert_called_once()

# --- Integration Tests ---

@pytest.mark.asyncio
async def test_full_interactive_flow_integration():
    """Test the full interactive flow integration."""
    with patch('builtins.input', side_effect=["learner_123", "1"]), \
         patch('src.app.pipeline') as mock_pipeline, \
         patch('builtins.print') as mock_print:
        
        # Mock all pipeline components
        mock_pipeline.vector_store_manager.get_weaviate_client.return_value = MagicMock()
        mock_pipeline.vector_store_manager.create_weaviate_schema.return_value = None
        
        mock_profile_manager = MagicMock()
        mock_profile_manager.close_db = MagicMock()
        mock_pipeline.profile_manager.LearnerProfileManager.return_value = mock_profile_manager
        
        mock_pipeline.retriever.Retriever.return_value = MagicMock()
        mock_pipeline.question_generator_rag.RAGQuestionGenerator.return_value = MagicMock()
        
        mock_question_selector = MagicMock()
        mock_question_selector.get_available_topics.return_value = [
            {"topic_id": "calculus", "source_file": "calculus.pdf"},
            {"topic_id": "algebra", "source_file": "algebra.pdf"}
        ]
        mock_pipeline.question_selector.QuestionSelector.return_value = mock_question_selector
        
        mock_pipeline.run_full_pipeline = AsyncMock()
        
        await main_interactive_app()
        
        # Verify the complete flow
        mock_pipeline.run_full_pipeline.assert_awaited_once_with(
            interactive_mode=True,
            initial_learner_id="learner_123",
            target_topic_id="calculus"
        )
        mock_profile_manager.close_db.assert_called_once() 