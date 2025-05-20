# tests/test_vector_store_manager.py
import unittest
from unittest.mock import patch, MagicMock, ANY
import uuid
import os # For os.getenv

# Mock config before other imports from src
# This is tricky if config is loaded at module level in other files.
# For robust testing, config might need to be injectable or lazily loaded.
# For now, we assume config is loaded when its module is imported.

from src import config # This will print config paths
# Temporarily suppress config prints for cleaner test output if desired
# with patch('builtins.print') as mock_print_config:
#     from src import config

from src.data_ingestion import vector_store_manager

# Global mock for SentenceTransformer to avoid actual model loading during tests
# This will be the return_value of the patched SentenceTransformer constructor
mock_st_model_instance = MagicMock()
mock_st_model_instance.get_sentence_embedding_dimension.return_value = 384 # Example dim
mock_st_model_instance.encode.return_value = [0.1] * 384 # Example embedding

@patch('src.data_ingestion.vector_store_manager.SentenceTransformer', return_value=mock_st_model_instance)
class TestVectorStoreManager(unittest.TestCase):

    def setUp(self):
        # Reset global embedding_model_instance in vector_store_manager for each test
        vector_store_manager.embedding_model_instance = None 
        
        # This is the mock instance that the patched weaviate.Client constructor will return
        self.client_mock_instance = MagicMock(spec=vector_store_manager.weaviate.Client)
        self.client_mock_instance.is_ready.return_value = True # Default ready state
        
        self.client_mock_instance.schema = MagicMock()
        # Default side effect for schema.get is "Not Found"
        self.client_mock_instance.schema.get.side_effect = vector_store_manager.weaviate.exceptions.UnexpectedStatusCodeException(status_code=404, message="Not Found")
        self.client_mock_instance.schema.create_class.return_value = None

        self.client_mock_instance.batch = MagicMock()
        self.client_mock_instance.batch.configure.return_value = None
        self.client_mock_instance.batch.add_data_object.return_value = None
        self.client_mock_instance.batch.failed_objects = [] 


    @patch('src.data_ingestion.vector_store_manager.weaviate.Client')
    def test_get_weaviate_client_success(self, mock_weaviate_constructor, mock_st_class_patch):
        mock_weaviate_constructor.return_value = self.client_mock_instance
        
        client = vector_store_manager.get_weaviate_client()
        self.assertIsNotNone(client)
        mock_weaviate_constructor.assert_called_with(url=config.WEAVIATE_URL)
        self.client_mock_instance.is_ready.assert_called()

    @patch('src.data_ingestion.vector_store_manager.weaviate.Client')
    @patch('time.sleep', return_value=None) # Mock time.sleep
    def test_get_weaviate_client_failure_then_success(self, mock_sleep, mock_weaviate_constructor, mock_st_class_patch):
        # Configure the mock instance that will be returned by the constructor
        self.client_mock_instance.is_ready.side_effect = [False, False, True] 
        mock_weaviate_constructor.return_value = self.client_mock_instance
        
        client = vector_store_manager.get_weaviate_client()
        self.assertIsNotNone(client)
        self.assertEqual(self.client_mock_instance.is_ready.call_count, 3)


    def test_create_weaviate_schema_new(self, mock_st_class_patch):
        # This test uses self.client_mock_instance directly
        vector_store_manager.create_weaviate_schema(self.client_mock_instance)
        
        self.client_mock_instance.schema.get.assert_called_with(vector_store_manager.WEAVIATE_CLASS_NAME)
        self.client_mock_instance.schema.create_class.assert_called_once()
        created_class_arg = self.client_mock_instance.schema.create_class.call_args[0][0]
        self.assertEqual(created_class_arg['class'], vector_store_manager.WEAVIATE_CLASS_NAME)
        self.assertIsNotNone(created_class_arg.get("vectorIndexConfig"))

    def test_create_weaviate_schema_exists(self, mock_st_class_patch):
        self.client_mock_instance.schema.get.side_effect = None # Clear the 404 side effect
        self.client_mock_instance.schema.get.return_value = {"class": vector_store_manager.WEAVIATE_CLASS_NAME} 
        
        vector_store_manager.create_weaviate_schema(self.client_mock_instance)
        self.client_mock_instance.schema.get.assert_called_with(vector_store_manager.WEAVIATE_CLASS_NAME)
        self.client_mock_instance.schema.create_class.assert_not_called()

    def test_generate_standard_embedding(self, mock_st_class_patch):
        # mock_st_class_patch refers to the class-level SentenceTransformer mock
        # get_embedding_model will use the mock_st_model_instance
        text = "This is a test sentence."
        embedding = vector_store_manager.generate_standard_embedding(text)
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), 384) 
        mock_st_model_instance.encode.assert_called_with(text, convert_to_tensor=False)

    def test_embed_chunk_data(self, mock_st_class_patch):
        chunk_data = {"chunk_text": "Sample chunk text."}
        embedding = vector_store_manager.embed_chunk_data(chunk_data)
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), 384)

    def test_embed_chunk_data_empty_text(self, mock_st_class_patch):
        chunk_data = {"chunk_text": ""}
        embedding = vector_store_manager.embed_chunk_data(chunk_data)
        self.assertIsNone(embedding)

    def test_embed_and_store_chunks(self, mock_st_class_patch):
        dummy_chunks = [
            {
                "chunk_id": str(uuid.uuid4()), "source": "s1", "original_type": "t1",
                "concept_type": "ct1", "concept_name": "cn1",
                "chunk_text": "text1", "parent_block_content": "pb1", "sequence_in_block": 0
            },
            {
                "chunk_id": str(uuid.uuid4()), "source": "s2", "original_type": "t2",
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
        with patch('src.data_ingestion.vector_store_manager.embed_chunk_data', side_effect=[[0.2]*384, None]) as mock_embed_chunk:
            dummy_chunks = [
                {"chunk_id": str(uuid.uuid4()), "chunk_text": "good text", "source": "s", "original_type": "t", "concept_type": "c", "parent_block_content": "p", "sequence_in_block": 0},
                {"chunk_id": str(uuid.uuid4()), "chunk_text": "bad text (simulated error)", "source": "s", "original_type": "t", "concept_type": "c", "parent_block_content": "p", "sequence_in_block": 0}
            ]
            vector_store_manager.embed_and_store_chunks(self.client_mock_instance, dummy_chunks)
            
            self.assertEqual(mock_embed_chunk.call_count, 2)
            self.client_mock_instance.batch.add_data_object.assert_called_once()


if __name__ == '__main__':
    unittest.main()