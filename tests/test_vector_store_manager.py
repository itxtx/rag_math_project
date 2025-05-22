# tests/test_vector_store_manager.py
import unittest
from unittest.mock import patch, MagicMock, ANY
import uuid
import os
import numpy as np 
import time 

from src import config 
from src.data_ingestion import vector_store_manager
import weaviate 
from sentence_transformers import SentenceTransformer 

mock_st_model_instance_vsm = MagicMock(spec=SentenceTransformer) 
mock_st_model_instance_vsm.get_sentence_embedding_dimension.return_value = 384 
mock_st_model_instance_vsm.encode.return_value = np.array([0.1] * 384, dtype=np.float32)

mock_sentence_transformer_class_vsm = MagicMock(spec=SentenceTransformer)
mock_sentence_transformer_class_vsm.return_value = mock_st_model_instance_vsm

@patch('src.data_ingestion.vector_store_manager.SentenceTransformer', new=mock_sentence_transformer_class_vsm)
class TestVectorStoreManager(unittest.TestCase):

    def setUp(self):
        vector_store_manager.embedding_model_instance = None 
        self.client_mock_instance = MagicMock(spec=weaviate.Client) 
        self.client_mock_instance.is_ready = MagicMock(return_value=True)
        self.client_mock_instance.schema = MagicMock()
        
        mock_404_response = MagicMock()
        mock_404_response.status_code = 404
        self.simulated_404_exception = weaviate.exceptions.UnexpectedStatusCodeException(
            message="Simulated Schema Not Found from setUp", response=mock_404_response
        )
        # Default for schema.get is to raise 404
        self.client_mock_instance.schema.get.side_effect = self.simulated_404_exception
        self.client_mock_instance.schema.exists = MagicMock(return_value=False) # Default: schema doesn't exist
        
        self.client_mock_instance.schema.create_class = MagicMock(return_value=None)
        self.client_mock_instance.schema.property = MagicMock() 
        self.client_mock_instance.schema.property.create = MagicMock()

        self.batch_mock = MagicMock() 
        self.batch_mock.configure = MagicMock(return_value=None)
        self.batch_mock.add_data_object = MagicMock(return_value=None)
        self.batch_mock.failed_objects = [] 
        
        self.client_mock_instance.batch = MagicMock() 
        self.client_mock_instance.batch.__enter__ = MagicMock(return_value=self.batch_mock) 
        self.client_mock_instance.batch.__exit__ = MagicMock(return_value=None)
        self.client_mock_instance.batch.failed_objects = []

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
        self.client_mock_instance.schema.exists.return_value = False # Ensure this test path
        # schema.get should not be called if exists is False.
        self.client_mock_instance.schema.get.reset_mock() # Reset any side_effect from setUp

        vector_store_manager.create_weaviate_schema(self.client_mock_instance)
        
        self.client_mock_instance.schema.exists.assert_called_with(vector_store_manager.WEAVIATE_CLASS_NAME)
        self.client_mock_instance.schema.get.assert_not_called() 
        self.client_mock_instance.schema.create_class.assert_called_once()
        # ... (rest of assertions)

    def test_create_weaviate_schema_exists(self):
        self.client_mock_instance.schema.exists.return_value = True
        self.client_mock_instance.schema.get.side_effect = None # Remove 404 side_effect
        self.client_mock_instance.schema.get.return_value = { 
            "class": vector_store_manager.WEAVIATE_CLASS_NAME,
            "properties": [
                {"name": p["name"]} for p in vector_store_manager.create_weaviate_schema.__defaults__[0]['properties'] # Simulate all props exist
            ] if vector_store_manager.create_weaviate_schema.__defaults__ else [] # Handle if no defaults
        }
        # If class_obj is directly accessible or can be constructed:
        # expected_props = [
        #     {"name": "chunk_id"}, {"name": "doc_id"}, {"name": "source_path"}, 
        #     {"name": "original_doc_type"}, {"name": "concept_type"}, {"name": "concept_name"},
        #     {"name": "chunk_text"}, {"name": "parent_block_id"}, {"name": "parent_block_content"},
        #     {"name": "sequence_in_block"}, {"name": "filename"}
        # ]
        # self.client_mock_instance.schema.get.return_value = {
        #     "class": vector_store_manager.WEAVIATE_CLASS_NAME,
        #     "properties": expected_props
        # }
        self.client_mock_instance.schema.property.create.reset_mock()

        vector_store_manager.create_weaviate_schema(self.client_mock_instance)

        self.client_mock_instance.schema.exists.assert_called_with(vector_store_manager.WEAVIATE_CLASS_NAME)
        self.client_mock_instance.schema.get.assert_called_with(vector_store_manager.WEAVIATE_CLASS_NAME) 
        self.client_mock_instance.schema.create_class.assert_not_called()
        self.client_mock_instance.schema.property.create.assert_not_called() 

    # ... (rest of the tests) ...
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
        self.assertEqual(self.batch_mock.add_data_object.call_count, 1)
        
        first_call_args = self.batch_mock.add_data_object.call_args_list[0]
        self.assertEqual(first_call_args[1]['class_name'], vector_store_manager.WEAVIATE_CLASS_NAME)
        self.assertEqual(first_call_args[1]['uuid'], dummy_chunks[0]['chunk_id'])
        self.assertIn('vector', first_call_args[1]) 
        self.assertEqual(len(first_call_args[1]['vector']), 384)
        self.assertEqual(first_call_args[1]['data_object']['chunk_text'], "text1")
        self.assertEqual(first_call_args[1]['data_object']['parent_block_id'], "pb1")


    def test_embed_and_store_chunks_embedding_error(self):
        self.client_mock_instance.schema.exists.return_value = False 
        self.client_mock_instance.schema.get.side_effect = self.simulated_404_exception

        with patch('src.data_ingestion.vector_store_manager.embed_chunk_data', side_effect=[np.array([0.2]*384).tolist(), None]) as mock_embed_func:
            dummy_chunks = [
                {"chunk_id": str(uuid.uuid4()), "chunk_text": "good text", "source_path": "s", "original_doc_type": "t", "concept_type": "c", "parent_block_content": "p", "sequence_in_block": 0, "parent_block_id": "pb_good", "doc_id":"doc_good"},
                {"chunk_id": str(uuid.uuid4()), "chunk_text": "bad text (simulated error)", "source_path": "s", "original_doc_type": "t", "concept_type": "c", "parent_block_content": "p", "sequence_in_block": 0, "parent_block_id": "pb_bad", "doc_id":"doc_bad"}
            ]
            vector_store_manager.embed_and_store_chunks(self.client_mock_instance, dummy_chunks)
            
            self.assertEqual(mock_embed_func.call_count, 2)
            self.batch_mock.add_data_object.assert_called_once() 


if __name__ == '__main__':
    unittest.main()
