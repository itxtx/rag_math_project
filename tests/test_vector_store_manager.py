# tests/test_vector_store_manager.py
import unittest
from unittest.mock import patch, MagicMock, ANY
import uuid
import os
import numpy as np # For mock embedding
import time # For sleep in one test

from src import config 
from src.data_ingestion import vector_store_manager
import weaviate # For weaviate.exceptions

# --- Mock setup for SentenceTransformer ---
# This is the instance our mocked SentenceTransformer class will return
mock_st_model_instance_vsm = MagicMock(spec=vector_store_manager.SentenceTransformer) # Use spec
mock_st_model_instance_vsm.get_sentence_embedding_dimension.return_value = 384 
mock_st_model_instance_vsm.encode.return_value = np.array([0.1] * 384, dtype=np.float32)

# This is a mock *class*. When SentenceTransformer() is called, this mock class will be called.
mock_sentence_transformer_class_vsm = MagicMock(spec=vector_store_manager.SentenceTransformer)
# Configure it so when it's called (instantiated), it returns our predefined model instance.
mock_sentence_transformer_class_vsm.return_value = mock_st_model_instance_vsm
# --- End of mock setup ---


# Patching SentenceTransformer with our mock_sentence_transformer_class using 'new='
@patch('src.data_ingestion.vector_store_manager.SentenceTransformer', new=mock_sentence_transformer_class_vsm)
class TestVectorStoreManager(unittest.TestCase):

    def setUp(self):
        vector_store_manager.embedding_model_instance = None 
        self.client_mock_instance = MagicMock(spec=weaviate.Client) 
        
        self.client_mock_instance.is_ready = MagicMock(return_value=True)
        
        # Schema mock setup
        self.client_mock_instance.schema = MagicMock()
        # Default side_effect for schema.get (can be overridden in specific tests)
        mock_404_response = MagicMock()
        mock_404_response.status_code = 404
        self.simulated_404_exception = weaviate.exceptions.UnexpectedStatusCodeException(
            message="Simulated Schema Not Found", response=mock_404_response
        )
        self.client_mock_instance.schema.get.side_effect = self.simulated_404_exception
        self.client_mock_instance.schema.exists = MagicMock(return_value=False) # For create_weaviate_schema
        self.client_mock_instance.schema.create_class = MagicMock(return_value=None)

        # Batch mock setup
        self.batch_mock = MagicMock() # This will be returned by client_mock_instance.batch
        self.batch_mock.configure = MagicMock(return_value=None)
        self.batch_mock.add_data_object = MagicMock(return_value=None)
        self.batch_mock.failed_objects = [] 
        
        # Configure client.batch to be a context manager returning self.batch_mock
        self.client_mock_instance.batch = MagicMock() # The main batch manager object
        self.client_mock_instance.batch.__enter__ = MagicMock(return_value=self.batch_mock) # __enter__ returns the object for 'as batch_context'
        self.client_mock_instance.batch.__exit__ = MagicMock(return_value=None)
        # Make client.batch.failed_objects accessible directly on client.batch after context
        self.client_mock_instance.batch.failed_objects = []


    # The mock_st_class_patch argument is no longer needed for test methods
    # because the class-level patch uses 'new=' and is active for all methods.
    @patch('src.data_ingestion.vector_store_manager.weaviate.Client')
    def test_get_weaviate_client_success(self, mock_weaviate_constructor):
        mock_weaviate_constructor.return_value = self.client_mock_instance 
        client = vector_store_manager.get_weaviate_client()
        self.assertIsNotNone(client)
        mock_weaviate_constructor.assert_called_with(url=config.WEAVIATE_URL)
        self.client_mock_instance.is_ready.assert_called()

    @patch('src.data_ingestion.vector_store_manager.weaviate.Client')
    @patch('time.sleep', return_value=None) 
    def test_get_weaviate_client_failure_then_success(self, mock_sleep, mock_weaviate_constructor):
        self.client_mock_instance.is_ready.side_effect = [False, False, True] 
        mock_weaviate_constructor.return_value = self.client_mock_instance
        
        client = vector_store_manager.get_weaviate_client()
        self.assertIsNotNone(client)
        self.assertEqual(self.client_mock_instance.is_ready.call_count, 3)

    def test_create_weaviate_schema_new(self):
        # schema.get will raise simulated_404_exception (set in setUp)
        # or we can ensure schema.exists returns False
        self.client_mock_instance.schema.exists.return_value = False
        self.client_mock_instance.schema.get.side_effect = self.simulated_404_exception # Keep for robustness

        vector_store_manager.create_weaviate_schema(self.client_mock_instance)
        
        # self.client_mock_instance.schema.get.assert_called_with(vector_store_manager.WEAVIATE_CLASS_NAME)
        self.client_mock_instance.schema.exists.assert_called_with(vector_store_manager.WEAVIATE_CLASS_NAME)
        self.client_mock_instance.schema.create_class.assert_called_once()
        created_class_arg = self.client_mock_instance.schema.create_class.call_args[0][0]
        self.assertEqual(created_class_arg['class'], vector_store_manager.WEAVIATE_CLASS_NAME)
        self.assertIn("parent_block_id", [p["name"] for p in created_class_arg["properties"]])


    def test_create_weaviate_schema_exists(self):
        # self.client_mock_instance.schema.get.side_effect = None 
        # self.client_mock_instance.schema.get.return_value = {"class": vector_store_manager.WEAVIATE_CLASS_NAME, "properties": [{"name": "parent_block_id"}]} 
        self.client_mock_instance.schema.exists.return_value = True
        # Mock .get() to return a schema that includes parent_block_id to avoid trying to add it
        self.client_mock_instance.schema.get.return_value = {
            "class": vector_store_manager.WEAVIATE_CLASS_NAME,
            "properties": [
                {"name": "chunk_id", "dataType": ["uuid"]},
                {"name": "parent_block_id", "dataType": ["text"]}, # Ensure it's in the mock schema
                # Add other essential properties if your update logic checks them
            ]
        }
        self.client_mock_instance.schema.property.create = MagicMock() # Mock property creation

        vector_store_manager.create_weaviate_schema(self.client_mock_instance)
        self.client_mock_instance.schema.exists.assert_called_with(vector_store_manager.WEAVIATE_CLASS_NAME)
        self.client_mock_instance.schema.create_class.assert_not_called()
        self.client_mock_instance.schema.property.create.assert_not_called() # Ensure no attempt to add if exists

    def test_generate_standard_embedding(self):
        text = "This is a test sentence."
        embedding = vector_store_manager.generate_standard_embedding(text)
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), 384) 
        mock_st_model_instance_vsm.encode.assert_called_with(text, convert_to_tensor=False, normalize_embeddings=True)

    def test_embed_chunk_data(self):
        chunk_data = {"chunk_text": "Sample chunk text."}
        embedding = vector_store_manager.embed_chunk_data(chunk_data)
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), 384)

    def test_embed_chunk_data_empty_text(self):
        chunk_data = {"chunk_text": "   "} 
        embedding = vector_store_manager.embed_chunk_data(chunk_data)
        self.assertIsNone(embedding)

    def test_embed_and_store_chunks(self):
        # Ensure schema.exists returns False so create_weaviate_schema tries to create it
        self.client_mock_instance.schema.exists.return_value = False
        self.client_mock_instance.schema.get.side_effect = self.simulated_404_exception


        dummy_chunks = [
            {
                "chunk_id": str(uuid.uuid4()), "doc_id": "d1", "source_path": "s1", "original_doc_type": "t1", 
                "concept_type": "ct1", "concept_name": "cn1", "parent_block_id": "pb1",
                "chunk_text": "text1", "parent_block_content": "pb_content1", "sequence_in_block": 0, "filename":"f1.tex"
            },
        ]
        
        vector_store_manager.embed_and_store_chunks(self.client_mock_instance, dummy_chunks, batch_size=1)
        
        self.client_mock_instance.batch.configure.assert_called_once_with(
            batch_size=1, dynamic=True, timeout_retries=3
        )
        # Check calls on the batch_mock (returned by __enter__)
        self.assertEqual(self.batch_mock.add_data_object.call_count, 1)
        
        first_call_args = self.batch_mock.add_data_object.call_args_list[0]
        self.assertEqual(first_call_args[1]['class_name'], vector_store_manager.WEAVIATE_CLASS_NAME)
        self.assertEqual(first_call_args[1]['uuid'], dummy_chunks[0]['chunk_id'])
        self.assertIn('vector', first_call_args[1]) 
        self.assertEqual(len(first_call_args[1]['vector']), 384)
        self.assertEqual(first_call_args[1]['data_object']['chunk_text'], "text1")
        self.assertEqual(first_call_args[1]['data_object']['parent_block_id'], "pb1")


    def test_embed_and_store_chunks_embedding_error(self):
        self.client_mock_instance.schema.exists.return_value = False # For schema creation path
        self.client_mock_instance.schema.get.side_effect = self.simulated_404_exception

        # Make embed_chunk_data return None for the second chunk
        with patch('src.data_ingestion.vector_store_manager.embed_chunk_data', side_effect=[np.array([0.2]*384).tolist(), None]) as mock_embed_func:
            dummy_chunks = [
                {"chunk_id": str(uuid.uuid4()), "chunk_text": "good text", "source_path": "s", "original_doc_type": "t", "concept_type": "c", "parent_block_content": "p", "sequence_in_block": 0, "parent_block_id": "pb_good", "doc_id":"doc_good"},
                {"chunk_id": str(uuid.uuid4()), "chunk_text": "bad text (simulated error)", "source_path": "s", "original_doc_type": "t", "concept_type": "c", "parent_block_content": "p", "sequence_in_block": 0, "parent_block_id": "pb_bad", "doc_id":"doc_bad"}
            ]
            vector_store_manager.embed_and_store_chunks(self.client_mock_instance, dummy_chunks)
            
            self.assertEqual(mock_embed_func.call_count, 2)
            self.batch_mock.add_data_object.assert_called_once() # Only one chunk should be added


if __name__ == '__main__':
    unittest.main()
