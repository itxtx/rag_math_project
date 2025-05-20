# tests/test_vector_store_manager.py
import unittest
from unittest.mock import patch, MagicMock, ANY
import uuid
import os 
import numpy as np
from src import config 
from src.data_ingestion import vector_store_manager

mock_st_model_instance = MagicMock()
mock_st_model_instance.get_sentence_embedding_dimension.return_value = 384
# Change this line:
mock_st_model_instance.encode.return_value = np.array([0.1] * 384, dtype=np.float32) # Return a NumPy array

@patch('src.data_ingestion.vector_store_manager.SentenceTransformer', return_value=mock_st_model_instance)
class TestVectorStoreManager(unittest.TestCase):

    def setUp(self):
        # Reset global embedding model instance for isolation if needed by tests
        vector_store_manager.embedding_model_instance = None

        # This is the mock instance that the patched weaviate.Client constructor will return
        self.client_mock_instance = MagicMock() # No spec initially to allow easier attribute assignment

        # Explicitly define methods/attributes expected to be called on the client instance
        self.client_mock_instance.is_ready = MagicMock(return_value=True)

        self.client_mock_instance.schema = MagicMock()

        # --- THIS IS THE MODIFIED SECTION ---
        # Create a mock response object that has a status_code attribute
        mock_response = MagicMock()
        mock_response.status_code = 404
        # mock_response.text = '{"error": "Not Found"}' # Optional: if your code uses response.text

        # Instantiate the exception with the message and the mock response
        simulated_404_exception = vector_store_manager.weaviate.exceptions.UnexpectedStatusCodeException(
            "Simulated Schema Not Found", # A descriptive message for the simulated error
            response=mock_response               # The mock response object
        )
        self.client_mock_instance.schema.get = MagicMock(side_effect=simulated_404_exception)
        # --- END OF MODIFIED SECTION ---

        self.client_mock_instance.schema.create_class = MagicMock(return_value=None)

        # Mocking the batch context manager parts
        # If client.batch is used as a context manager `with client.batch as batch:`
        # self.client_mock_instance.batch.__enter__ = MagicMock(return_value=self.client_mock_instance.batch)
        # self.client_mock_instance.batch.__exit__ = MagicMock(return_value=None)
        # Or simpler if batch itself is just an object with methods:
        self.client_mock_instance.batch = MagicMock()
        self.client_mock_instance.batch.configure = MagicMock(return_value=None)
        self.client_mock_instance.batch.add_data_object = MagicMock(return_value=None)
        self.client_mock_instance.batch.failed_objects = [] # Ensure it's an empty list for tests not expecting failures
        self.client_mock_instance.batch.__enter__ = MagicMock(return_value=self.client_mock_instance.batch)
        self.client_mock_instance.batch.__exit__ = MagicMock(return_value=None) # For completeness


    @patch('src.data_ingestion.vector_store_manager.weaviate.Client')
    def test_get_weaviate_client_success(self, mock_weaviate_constructor, mock_st_class_patch):
        mock_weaviate_constructor.return_value = self.client_mock_instance # Constructor returns our mock
        
        client = vector_store_manager.get_weaviate_client()
        self.assertIsNotNone(client)
        mock_weaviate_constructor.assert_called_with(url=config.WEAVIATE_URL)
        client.is_ready.assert_called() # is_ready was called on the instance

    @patch('src.data_ingestion.vector_store_manager.weaviate.Client')
    @patch('time.sleep', return_value=None) 
    def test_get_weaviate_client_failure_then_success(self, mock_sleep, mock_weaviate_constructor, mock_st_class_patch):
        # Configure the specific behavior for is_ready on our instance
        self.client_mock_instance.is_ready.side_effect = [False, False, True] 
        mock_weaviate_constructor.return_value = self.client_mock_instance
        
        client = vector_store_manager.get_weaviate_client()
        self.assertIsNotNone(client)
        self.assertEqual(client.is_ready.call_count, 3)


    def test_create_weaviate_schema_new(self, mock_st_class_patch):
        # This test uses self.client_mock_instance directly, which is fine.
        vector_store_manager.create_weaviate_schema(self.client_mock_instance)
        
        self.client_mock_instance.schema.get.assert_called_with(vector_store_manager.WEAVIATE_CLASS_NAME)
        self.client_mock_instance.schema.create_class.assert_called_once()
        created_class_arg = self.client_mock_instance.schema.create_class.call_args[0][0]
        self.assertEqual(created_class_arg['class'], vector_store_manager.WEAVIATE_CLASS_NAME)
        self.assertIsNotNone(created_class_arg.get("vectorIndexConfig"))

    def test_create_weaviate_schema_exists(self, mock_st_class_patch):
        self.client_mock_instance.schema.get.side_effect = None 
        self.client_mock_instance.schema.get.return_value = {"class": vector_store_manager.WEAVIATE_CLASS_NAME} 
        
        vector_store_manager.create_weaviate_schema(self.client_mock_instance)
        self.client_mock_instance.schema.get.assert_called_with(vector_store_manager.WEAVIATE_CLASS_NAME)
        self.client_mock_instance.schema.create_class.assert_not_called()

    def test_generate_standard_embedding(self, mock_st_class_patch):
        text = "This is a test sentence."
        embedding = vector_store_manager.generate_standard_embedding(text)
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), 384) 
        mock_st_model_instance.encode.assert_called_with(text, convert_to_tensor=False, normalize_embeddings=True)

    def test_embed_chunk_data(self, mock_st_class_patch):
        chunk_data = {"chunk_text": "Sample chunk text."}
        embedding = vector_store_manager.embed_chunk_data(chunk_data)
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), 384)

    def test_embed_chunk_data_empty_text(self, mock_st_class_patch):
        chunk_data = {"chunk_text": "  "} 
        embedding = vector_store_manager.embed_chunk_data(chunk_data)
        self.assertIsNone(embedding)

    def test_embed_and_store_chunks(self, mock_st_class_patch):
        dummy_chunks = [
            {
                "chunk_id": str(uuid.uuid4()), "source_path": "s1", "original_doc_type": "t1", # Use new field names
                "concept_type": "ct1", "concept_name": "cn1",
                "chunk_text": "text1", "parent_block_content": "pb1", "sequence_in_block": 0
            },
            {
                "chunk_id": str(uuid.uuid4()), "source_path": "s2", "original_doc_type": "t2", # Use new field names
                "concept_type": "ct2", "concept_name": "cn2",
                "chunk_text": "text2", "parent_block_content": "pb2", "sequence_in_block": 0
            }
        ]
        
        vector_store_manager.embed_and_store_chunks(self.client_mock_instance, dummy_chunks, batch_size=1)
        
        self.client_mock_instance.batch.configure.assert_called_once_with(
            batch_size=1, dynamic=True, timeout_retries=3
        )
        self.assertEqual(self.client_mock_instance.batch.add_data_object.call_count, 2)
        
        first_call_args = self.client_mock_instance.batch.add_data_object.call_args_list[0]
        self.assertEqual(first_call_args[1]['class_name'], vector_store_manager.WEAVIATE_CLASS_NAME)
        self.assertEqual(first_call_args[1]['uuid'], dummy_chunks[0]['chunk_id'])
        self.assertIn('vector', first_call_args[1]) 
        self.assertEqual(len(first_call_args[1]['vector']), 384)
        self.assertEqual(first_call_args[1]['data_object']['chunk_text'], "text1")

    def test_embed_and_store_chunks_embedding_error(self, mock_st_class_patch):
        with patch('src.data_ingestion.vector_store_manager.embed_chunk_data', side_effect=[[0.2]*384, None]) as mock_embed_chunk_func:
            dummy_chunks = [
                {"chunk_id": str(uuid.uuid4()), "chunk_text": "good text", "source_path": "s", "original_doc_type": "t", "concept_type": "c", "parent_block_content": "p", "sequence_in_block": 0},
                {"chunk_id": str(uuid.uuid4()), "chunk_text": "bad text (simulated error)", "source_path": "s", "original_doc_type": "t", "concept_type": "c", "parent_block_content": "p", "sequence_in_block": 0}
            ]
            vector_store_manager.embed_and_store_chunks(self.client_mock_instance, dummy_chunks)
            
            self.assertEqual(mock_embed_chunk_func.call_count, 2)
            self.client_mock_instance.batch.add_data_object.assert_called_once()


if __name__ == '__main__':
    unittest.main()