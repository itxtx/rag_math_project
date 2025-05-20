
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import uuid
import asyncio # For running async methods
import time
import numpy as np

# Modules to test and helpers
from src.retrieval.retriever import Retriever
from src.generation.question_generator_rag import RAGQuestionGenerator
from src.data_ingestion import vector_store_manager
from src import config # For Weaviate URL, class name etc.
import weaviate # For specific exceptions
from sentence_transformers import SentenceTransformer # Import for spec

# --- Mock setup for SentenceTransformer ---
# This is the instance our mocked SentenceTransformer class will return
mock_st_model_instance_integration = MagicMock(spec=SentenceTransformer) # Use spec for better mocking
mock_st_model_instance_integration.get_sentence_embedding_dimension.return_value = 384
mock_st_model_instance_integration.encode.return_value = np.array([0.1] * 384, dtype=np.float32)

# This is a mock *class*. When SentenceTransformer() is called, this mock class will be called.
mock_sentence_transformer_class = MagicMock(spec=SentenceTransformer)
# Configure it so when it's called (instantiated), it returns our predefined model instance.
mock_sentence_transformer_class.return_value = mock_st_model_instance_integration
# --- End of mock setup ---


# Test data constants
TEST_CHUNK_1_ID = str(uuid.uuid4())
TEST_CHUNK_1_TEXT = "The first law of thermodynamics, also known as the law of conservation of energy, states that energy cannot be created or destroyed in an isolated system. It can only be transformed from one form to another."
TEST_CHUNK_1_CONCEPT = "First Law of Thermodynamics"

TEST_CHUNK_2_ID = str(uuid.uuid4())
TEST_CHUNK_2_TEXT = "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy, through a process that uses sunlight, water, and carbon dioxide."
TEST_CHUNK_2_CONCEPT = "Photosynthesis"

DUMMY_CHUNKS_FOR_INTEGRATION_TEST = [
    {
        "chunk_id": TEST_CHUNK_1_ID, "source_path": "thermo_doc.tex", "original_doc_type": "latex",
        "concept_type": "scientific_law", "concept_name": TEST_CHUNK_1_CONCEPT,
        "chunk_text": TEST_CHUNK_1_TEXT,
        "parent_block_content": f"\\section{{{TEST_CHUNK_1_CONCEPT}}}\n{TEST_CHUNK_1_TEXT} This law is fundamental to understanding energy transformations.",
        "sequence_in_block": 0
    },
    {
        "chunk_id": TEST_CHUNK_2_ID, "source_path": "biology_notes.pdf", "original_doc_type": "pdf",
        "concept_type": "biological_process", "concept_name": TEST_CHUNK_2_CONCEPT,
        "chunk_text": TEST_CHUNK_2_TEXT,
        "parent_block_content": f"Chapter on {TEST_CHUNK_2_CONCEPT}.\n{TEST_CHUNK_2_TEXT} This process is vital for life on Earth.",
        "sequence_in_block": 0
    }
]

# Patching SentenceTransformer with our mock_sentence_transformer_class.
# 'new' replaces the original class with our mock class.
@patch('src.data_ingestion.vector_store_manager.SentenceTransformer', new=mock_sentence_transformer_class)
class TestRetrievalAndGenerationIntegration(unittest.TestCase):

    weaviate_client = None

    @classmethod
    def setUpClass(cls):
        """Set up for all tests; connect to Weaviate and ingest test data."""
        print("\n--- Integration Test Setup (setUpClass) ---")
        # The class-level patch with 'new=mock_sentence_transformer_class' is active.
        # So, when vector_store_manager.get_embedding_model() calls SentenceTransformer(...),
        # our mock_sentence_transformer_class is called, which returns mock_st_model_instance_integration.
        # This should prevent actual model downloads.
        try:
            if not hasattr(config, 'WEAVIATE_URL') or not config.WEAVIATE_URL:
                raise unittest.SkipTest("WEAVIATE_URL not configured in src.config; skipping integration tests.")

            cls.weaviate_client = vector_store_manager.get_weaviate_client()
            print(f"Weaviate client obtained for class: {vector_store_manager.WEAVIATE_CLASS_NAME}")

            vector_store_manager.create_weaviate_schema(cls.weaviate_client)
            print("Schema ensured.")

            print(f"Ingesting {len(DUMMY_CHUNKS_FOR_INTEGRATION_TEST)} dummy chunks for integration test...")
            vector_store_manager.embed_and_store_chunks(
                cls.weaviate_client,
                DUMMY_CHUNKS_FOR_INTEGRATION_TEST,
                batch_size=2
            )
            print("Dummy chunks ingestion call completed.")
            print("Waiting for Weaviate to index...")
            time.sleep(2)
            print("Indexing wait finished.")

        except ConnectionRefusedError as e:
            print(f"CRITICAL ERROR in setUpClass: Weaviate connection refused at {config.WEAVIATE_URL}. {e}")
            print("Ensure Weaviate is running and accessible.")
            raise unittest.SkipTest(f"Weaviate connection refused. Skipping integration tests. {e}")
        except Exception as e:
            print(f"CRITICAL ERROR in setUpClass: {e}")
            import traceback
            traceback.print_exc()
            raise unittest.SkipTest(f"Critical error in setUpClass, skipping integration tests. {e}")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests; remove test data."""
        print("\n--- Integration Test Teardown (tearDownClass) ---")
        if hasattr(cls, 'weaviate_client') and cls.weaviate_client:
            try:
                print("Attempting to delete test objects...")
                deleted_count = 0
                for chunk_data in DUMMY_CHUNKS_FOR_INTEGRATION_TEST:
                    try:
                        cls.weaviate_client.data_object.delete(
                            uuid=chunk_data["chunk_id"],
                            class_name=vector_store_manager.WEAVIATE_CLASS_NAME
                        )
                        print(f"Deleted object with UUID: {chunk_data['chunk_id']}")
                        deleted_count +=1
                    except weaviate.exceptions.UnexpectedStatusCodeException as e:
                        if e.status_code == 404:
                            print(f"Object {chunk_data['chunk_id']} not found for deletion.")
                        else:
                            print(f"Warning: Could not delete object {chunk_data['chunk_id']}. Status {e.status_code}, Error: {e}")
                    except Exception as e_general:
                         print(f"Warning: General error deleting object {chunk_data['chunk_id']}: {e_general}")
                print(f"Test objects deletion process finished. {deleted_count} objects targeted for deletion.")
            except Exception as e:
                print(f"Error during tearDownClass: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Weaviate client not available in tearDownClass or was not initialized.")

    # The mock_st_vsm_method_level argument is not needed here anymore if the class patch uses 'new'.
    # The patch is globally active for the class.
    def test_retrieve_and_generate_questions_flow(self): # Removed the mock argument
        """
        Tests the end-to-end flow: retrieve chunks from Weaviate, then generate questions.
        """
        print("\nRunning test_retrieve_and_generate_questions_flow...")
        self.assertIsNotNone(self.weaviate_client, "Weaviate client not initialized in setUpClass.")

        retriever = Retriever(weaviate_client=self.weaviate_client)
        self.assertIsNotNone(retriever.client, "Retriever's client should be initialized.")

        query = "energy conservation law"
        print(f"Retrieving chunks for query: '{query}'")
        retrieved_chunks = retriever.search(query_text=query, search_type="semantic", limit=1, certainty=0.01)

        self.assertIsNotNone(retrieved_chunks, "retrieved_chunks should not be None.")
        self.assertGreater(len(retrieved_chunks), 0, f"Should retrieve at least one chunk for query '{query}'. Found {len(retrieved_chunks)}.")
        
        retrieved_texts = [chunk.get("chunk_text", "") for chunk in retrieved_chunks]
        print(f"Retrieved texts for query '{query}': {retrieved_texts}")
        self.assertTrue(
            any(TEST_CHUNK_1_TEXT in text for text in retrieved_texts),
            f"Expected text related to '{TEST_CHUNK_1_CONCEPT}' in retrieved chunks. Actual: {retrieved_texts}"
        )

        question_generator = RAGQuestionGenerator()
        mock_llm_response_text = "1. What does the first law of thermodynamics state about energy?\n2. Can energy be created or destroyed according to this law?"
        question_generator._call_llm_api = AsyncMock(return_value=mock_llm_response_text)

        print("Generating questions from retrieved chunks...")
        num_questions_to_generate = 2
        
        async def run_async_generate_questions():
            return await question_generator.generate_questions(
                context_chunks=retrieved_chunks,
                num_questions=num_questions_to_generate,
                question_type="factual"
            )
        generated_questions = asyncio.run(run_async_generate_questions())

        self.assertIsNotNone(generated_questions, "Generated questions list should not be None.")
        self.assertEqual(len(generated_questions), num_questions_to_generate,
                         f"Expected {num_questions_to_generate} questions, got {len(generated_questions)}: {generated_questions}")
        
        expected_questions = [
            "What does the first law of thermodynamics state about energy?",
            "Can energy be created or destroyed according to this law?"
        ]
        self.assertListEqual(generated_questions, expected_questions,
                             f"Generated questions do not match expected. Got: {generated_questions}")

        question_generator._call_llm_api.assert_called_once()
        call_args = question_generator._call_llm_api.call_args
        self.assertIsNotNone(call_args)
        prompt_sent_to_llm = call_args[0][0]
        
        self.assertIn(TEST_CHUNK_1_TEXT, prompt_sent_to_llm)
        self.assertIn(f"Generate exactly {num_questions_to_generate} questions", prompt_sent_to_llm)
        self.assertIn("factual questions", prompt_sent_to_llm)
        print("Assertions for generated questions and LLM call passed.")

if __name__ == '__main__':
    unittest.main()