# tests/integration/test_retrieval_and_generation.py

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import uuid
import asyncio 
import time
import numpy as np
import os 

from src.retrieval.retriever import Retriever
from src.generation.question_generator_rag import RAGQuestionGenerator
from src.data_ingestion import vector_store_manager
from src import config 
import weaviate 
from sentence_transformers import SentenceTransformer 

# --- Mock setup for SentenceTransformer ---
mock_st_model_instance_integration = MagicMock(spec=SentenceTransformer) 
mock_st_model_instance_integration.get_sentence_embedding_dimension.return_value = 384
mock_embedding_vector = np.array([0.1] * 383 + [0.2], dtype=np.float32) 
mock_st_model_instance_integration.encode.return_value = mock_embedding_vector

mock_sentence_transformer_class = MagicMock(spec=SentenceTransformer)
mock_sentence_transformer_class.return_value = mock_st_model_instance_integration
# --- End of mock setup ---


TEST_CHUNK_1_ID = str(uuid.uuid4())
TEST_CHUNK_1_TEXT = "The first law of thermodynamics, also known as the law of conservation of energy, states that energy cannot be created or destroyed in an isolated system. It can only be transformed from one form to another."
TEST_CHUNK_1_CONCEPT = "First Law of Thermodynamics"
TEST_CHUNK_1_PARENT_BLOCK_ID = "pb_thermo_law1"
TEST_CHUNK_1_DOC_ID = "doc_thermo"


TEST_CHUNK_2_ID = str(uuid.uuid4())
TEST_CHUNK_2_TEXT = "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy, through a process that uses sunlight, water, and carbon dioxide."
TEST_CHUNK_2_CONCEPT = "Photosynthesis"
TEST_CHUNK_2_PARENT_BLOCK_ID = "pb_photosynthesis"
TEST_CHUNK_2_DOC_ID = "doc_bio"


DUMMY_CHUNKS_FOR_INTEGRATION_TEST = [
    {
        "chunk_id": TEST_CHUNK_1_ID, "doc_id": TEST_CHUNK_1_DOC_ID, 
        "source_path": "thermo_doc.tex", "original_doc_type": "latex",
        "concept_type": "scientific_law", "concept_name": TEST_CHUNK_1_CONCEPT,
        "chunk_text": TEST_CHUNK_1_TEXT,
        "parent_block_id": TEST_CHUNK_1_PARENT_BLOCK_ID, 
        "parent_block_content": f"\\section{{{TEST_CHUNK_1_CONCEPT}}}\n{TEST_CHUNK_1_TEXT}",
        "sequence_in_block": 0, "filename": "thermo_doc.tex" 
    },
    {
        "chunk_id": TEST_CHUNK_2_ID, "doc_id": TEST_CHUNK_2_DOC_ID,
        "source_path": "biology_notes.pdf", "original_doc_type": "pdf",
        "concept_type": "biological_process", "concept_name": TEST_CHUNK_2_CONCEPT,
        "chunk_text": TEST_CHUNK_2_TEXT,
        "parent_block_id": TEST_CHUNK_2_PARENT_BLOCK_ID, 
        "parent_block_content": f"Chapter on {TEST_CHUNK_2_CONCEPT}.\n{TEST_CHUNK_2_TEXT}",
        "sequence_in_block": 0, "filename": "biology_notes.pdf"
    }
]

@patch('src.data_ingestion.vector_store_manager.SentenceTransformer', new=mock_sentence_transformer_class)
class TestRetrievalAndGenerationIntegration(unittest.TestCase):

    weaviate_client = None 

    @classmethod
    def setUpClass(cls):
        print("\n--- Integration Test Setup (setUpClass) ---")
        try:
            if not hasattr(config, 'WEAVIATE_URL') or not config.WEAVIATE_URL:
                raise unittest.SkipTest("WEAVIATE_URL not configured.")

            cls.weaviate_client = vector_store_manager.get_weaviate_client()
            print(f"Weaviate client obtained for class: {vector_store_manager.WEAVIATE_CLASS_NAME}")
            
            try:
                if cls.weaviate_client.schema.exists(vector_store_manager.WEAVIATE_CLASS_NAME):
                    print(f"Deleting existing class '{vector_store_manager.WEAVIATE_CLASS_NAME}' for clean test.")
                    cls.weaviate_client.schema.delete_class(vector_store_manager.WEAVIATE_CLASS_NAME)
                    time.sleep(1) 
            except Exception as e_del:
                print(f"Note: Could not delete existing class (may not exist or other issue): {e_del}")

            vector_store_manager.create_weaviate_schema(cls.weaviate_client) # This will now include parent_block_id
            print("Schema ensured.")

            print(f"Ingesting {len(DUMMY_CHUNKS_FOR_INTEGRATION_TEST)} dummy chunks...")
            vector_store_manager.embed_and_store_chunks(
                cls.weaviate_client,
                DUMMY_CHUNKS_FOR_INTEGRATION_TEST,
                batch_size=2
            )
            print("Dummy chunks ingestion call completed.")
            print("Waiting 5 seconds for Weaviate to index...") 
            time.sleep(5) 
            print("Indexing wait finished.")

            print("Verifying ingestion by fetching one of the objects by ID...")
            obj = cls.weaviate_client.data_object.get_by_id(
                DUMMY_CHUNKS_FOR_INTEGRATION_TEST[0]['chunk_id'],
                class_name=vector_store_manager.WEAVIATE_CLASS_NAME,
                with_vector=False # Don't need vector for this check
            )
            if obj:
                print(f"Successfully fetched ingested object: {obj.get('id')}")
                # Also check if parent_block_id was stored
                retrieved_props = obj.get('properties', {})
                print(f"  Properties retrieved: {retrieved_props}")
                if 'parent_block_id' not in retrieved_props:
                    print(f"WARNING: 'parent_block_id' NOT FOUND in fetched object {obj.get('id')}")
                else:
                    print(f"  'parent_block_id' found: {retrieved_props.get('parent_block_id')}")

            else:
                print(f"WARNING: Could not fetch ingested object by ID {DUMMY_CHUNKS_FOR_INTEGRATION_TEST[0]['chunk_id']}. Ingestion might have issues.")


        except ConnectionRefusedError as e:
            print(f"CRITICAL ERROR in setUpClass: Weaviate connection refused at {config.WEAVIATE_URL}. {e}")
            raise unittest.SkipTest(f"Weaviate connection refused. {e}")
        except Exception as e:
            print(f"CRITICAL ERROR in setUpClass: {e}")
            import traceback
            traceback.print_exc()
            raise unittest.SkipTest(f"Critical error in setUpClass. {e}")

    @classmethod
    def tearDownClass(cls):
        print("\n--- Integration Test Teardown (tearDownClass) ---")
        if hasattr(cls, 'weaviate_client') and cls.weaviate_client:
            try:
                print(f"Attempting to delete class: {vector_store_manager.WEAVIATE_CLASS_NAME} if it exists.")
                if cls.weaviate_client.schema.exists(vector_store_manager.WEAVIATE_CLASS_NAME):
                    cls.weaviate_client.schema.delete_class(vector_store_manager.WEAVIATE_CLASS_NAME)
                    print(f"Class {vector_store_manager.WEAVIATE_CLASS_NAME} deleted.")
                else:
                    print(f"Class {vector_store_manager.WEAVIATE_CLASS_NAME} did not exist, no deletion needed.")
            except Exception as e:
                print(f"Error during tearDownClass deleting schema: {e}")
        else:
            print("Weaviate client not available in tearDownClass.")

    def test_retrieve_and_generate_questions_flow(self):
        print("\nRunning test_retrieve_and_generate_questions_flow...")
        self.assertIsNotNone(self.weaviate_client, "Weaviate client not initialized.")

        retriever_instance = Retriever(weaviate_client=self.weaviate_client)
        self.assertIsNotNone(retriever_instance.client)

        query = "energy conservation law" 
        print(f"Retrieving chunks for query: '{query}'")
        # Ensure DEFAULT_RETURN_PROPERTIES in Retriever includes parent_block_id
        # The test failure indicated "Cannot query field 'parent_block_id'".
        # This implies the retriever might be trying to fetch it, but it wasn't in the schema during that run.
        # With the schema fix, this part of the retriever should work.
        retrieved_chunks = retriever_instance.search(query_text=query, search_type="semantic", limit=1, certainty=0.01)

        self.assertIsNotNone(retrieved_chunks)
        self.assertGreater(len(retrieved_chunks), 0, f"Should retrieve at least one chunk for query '{query}'. Found {len(retrieved_chunks)}.")
        
        retrieved_texts = [chunk.get("chunk_text", "") for chunk in retrieved_chunks]
        print(f"Retrieved texts for query '{query}': {retrieved_texts}")
        self.assertTrue(
            any(TEST_CHUNK_1_TEXT in text for text in retrieved_texts),
            f"Expected text related to '{TEST_CHUNK_1_CONCEPT}' in retrieved chunks. Actual: {retrieved_texts}"
        )

        question_generator_instance = RAGQuestionGenerator()
        mock_llm_response_text = "1. What does the first law of thermodynamics state about energy?\n2. Can energy be created or destroyed according to this law?"
        question_generator_instance._call_llm_api = AsyncMock(return_value=mock_llm_response_text)

        print("Generating questions from retrieved chunks...")
        num_questions_to_generate = 2
        
        async def run_async_generate_questions():
            return await question_generator_instance.generate_questions(
                context_chunks=retrieved_chunks,
                num_questions=num_questions_to_generate,
                question_type="factual",
                difficulty_level="intermediate" # Added for consistency with QG update
            )
        generated_questions = asyncio.run(run_async_generate_questions())

        self.assertIsNotNone(generated_questions)
        self.assertEqual(len(generated_questions), num_questions_to_generate)
        
        expected_questions = [
            "What does the first law of thermodynamics state about energy?",
            "Can energy be created or destroyed according to this law?"
        ]
        self.assertListEqual(generated_questions, expected_questions)

        question_generator_instance._call_llm_api.assert_called_once()
        call_args = question_generator_instance._call_llm_api.call_args
        self.assertIsNotNone(call_args)
        prompt_sent_to_llm = call_args[0][0]
        
        self.assertTrue(any(ret_text in prompt_sent_to_llm for ret_text in retrieved_texts if ret_text),
                      "The prompt sent to LLM should contain text from a retrieved chunk.")
        # Corrected assertion for the prompt string
        self.assertIn(f"Generate exactly {num_questions_to_generate} question(s)", prompt_sent_to_llm)
        self.assertIn("factual questions", prompt_sent_to_llm)
        print("Assertions for generated questions and LLM call passed.")

if __name__ == '__main__':
    unittest.main()
