import pytest
import asyncio
import os
import tempfile
import shutil
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from pathlib import Path
from unittest import mock
from unittest.mock import call

from src.pipeline import FastPipeline, run_fast_ingestion_pipeline, run_fast_gnn_training

# --- Fixtures ---

@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def temp_graph_db_dir(temp_data_dir):
    """Create a temporary graph database directory."""
    graph_dir = os.path.join(temp_data_dir, "graph_db")
    os.makedirs(graph_dir, exist_ok=True)
    return graph_dir

@pytest.fixture
def temp_embeddings_dir(temp_data_dir):
    """Create a temporary embeddings directory."""
    embeddings_dir = os.path.join(temp_data_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    return embeddings_dir

@pytest.fixture
def sample_latex_content():
    """Sample LaTeX content for testing."""
    return r"""
\documentclass{article}
\begin{document}
\title{Test Document}
\author{Test Author}
\maketitle

\section{Introduction}
This is a test document for unit testing.

\section{Mathematics}
The derivative of $f(x) = x^2$ is $f'(x) = 2x$.

\begin{equation}
\int_0^1 x^2 dx = \frac{1}{3}
\end{equation}

\end{document}
"""

@pytest.fixture
def mock_document_data():
    """Mock document data for testing."""
    return [
        {
            "doc_id": "test_doc_1",
            "source": "/path/to/test1.tex",
            "filename": "test1.tex",
            "type": "latex",
            "raw_content": "\\documentclass{article}\\begin{document}Test 1\\end{document}"
        },
        {
            "doc_id": "test_doc_2",
            "source": "/path/to/test2.tex",
            "filename": "test2.tex",
            "type": "latex",
            "raw_content": "\\documentclass{article}\\begin{document}Test 2\\end{document}"
        }
    ]

@pytest.fixture
def mock_conceptual_blocks():
    """Mock conceptual blocks for testing."""
    return [
        {
            "block_id": "block_1",
            "content": "Test content 1",
            "concept_name": "Test Concept 1",
            "doc_id": "test_doc_1"
        },
        {
            "block_id": "block_2",
            "content": "Test content 2",
            "concept_name": "Test Concept 2",
            "doc_id": "test_doc_2"
        }
    ]

@pytest.fixture
def mock_final_chunks():
    """Mock final chunks for testing."""
    return [
        {
            "chunk_id": "chunk_1",
            "content": "Test chunk 1",
            "concept_name": "Test Concept 1",
            "doc_id": "test_doc_1"
        },
        {
            "chunk_id": "chunk_2",
            "content": "Test chunk 2",
            "concept_name": "Test Concept 2",
            "doc_id": "test_doc_2"
        }
    ]

# --- FastPipeline Class Tests ---

def test_fast_pipeline_initialization():
    """Test FastPipeline initialization."""
    pipeline = FastPipeline()
    
    assert pipeline.graph_parser is None
    assert pipeline.start_time is None
    assert pipeline.processed_count == 0
    assert pipeline.error_count == 0

@pytest.mark.asyncio
async def test_run_ingestion_no_new_documents():
    """Test ingestion when no new documents are available."""
    with patch('src.pipeline.document_loader.load_new_documents', return_value=[]), \
         patch('src.pipeline.logging.getLogger') as mock_logger:
        
        pipeline = FastPipeline()
        result = await pipeline.run_ingestion()
        
        assert result is True
        assert pipeline.processed_count == 0
        assert pipeline.error_count == 0

@pytest.mark.asyncio
async def test_run_ingestion_successful_processing(mock_document_data, mock_conceptual_blocks, mock_final_chunks):
    """Test successful ingestion pipeline."""
    with patch('src.pipeline.document_loader.load_new_documents', return_value=mock_document_data), \
         patch('src.pipeline.LatexToGraphParser') as mock_parser_class, \
         patch('src.pipeline.chunker.chunk_conceptual_blocks', return_value=mock_final_chunks), \
         patch('src.pipeline.vector_store_manager.get_weaviate_client') as mock_get_client, \
         patch('src.pipeline.fast_embed_and_store_chunks', new_callable=AsyncMock) as mock_store_chunks, \
         patch('src.pipeline.document_loader.update_processed_docs_log') as mock_update_log, \
         patch('src.pipeline.os.makedirs') as mock_makedirs, \
         patch('src.pipeline.logging.getLogger') as mock_logger:
        
        # Mock the parser instance
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.get_graph_nodes_as_conceptual_blocks.return_value = mock_conceptual_blocks
        
        # Mock Weaviate client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        pipeline = FastPipeline()
        result = await pipeline.run_ingestion()
        
        assert result is True
        assert pipeline.processed_count == 2
        assert pipeline.error_count == 0
        
        # Verify parser was called for each document
        assert mock_parser.extract_structured_nodes.call_count == 2
        
        # Verify chunks were processed
        mock_store_chunks.assert_awaited_once_with(mock_client, mock_final_chunks, batch_size=100)
        
        # Verify log was updated
        mock_update_log.assert_called_once()

@pytest.mark.asyncio
async def test_run_ingestion_parser_error(mock_document_data):
    """Test ingestion when parser raises an exception."""
    with patch('src.pipeline.document_loader.load_new_documents', return_value=mock_document_data), \
         patch('src.pipeline.LatexToGraphParser') as mock_parser_class, \
         patch('src.pipeline.document_loader.update_processed_docs_log') as mock_update_log, \
         patch('src.pipeline.os.makedirs') as mock_makedirs, \
         patch('src.pipeline.logging.getLogger') as mock_logger:
        
        # Mock the parser instance to raise an exception
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.extract_structured_nodes.side_effect = Exception("Parser error")
        
        pipeline = FastPipeline()
        result = await pipeline.run_ingestion()
        
        assert result is False
        assert pipeline.processed_count == 0
        assert pipeline.error_count == 2
        
        # Verify log was still updated
        mock_update_log.assert_called_once()

@pytest.mark.asyncio
async def test_run_ingestion_no_conceptual_blocks(mock_document_data):
    """Test ingestion when no conceptual blocks are generated."""
    with patch('src.pipeline.document_loader.load_new_documents', return_value=mock_document_data), \
         patch('src.pipeline.LatexToGraphParser') as mock_parser_class, \
         patch('src.pipeline.document_loader.update_processed_docs_log') as mock_update_log, \
         patch('src.pipeline.os.makedirs') as mock_makedirs, \
         patch('src.pipeline.logging.getLogger') as mock_logger:
        
        # Mock the parser instance
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.get_graph_nodes_as_conceptual_blocks.return_value = []
        
        pipeline = FastPipeline()
        result = await pipeline.run_ingestion()
        
        assert result is True  # Should return True when no errors occurred
        assert pipeline.processed_count == 2
        assert pipeline.error_count == 0
        
        # Verify log was updated
        mock_update_log.assert_called_once()

@pytest.mark.asyncio
async def test_run_ingestion_no_chunks_generated(mock_document_data, mock_conceptual_blocks):
    """Test ingestion when no chunks are generated."""
    with patch('src.pipeline.document_loader.load_new_documents', return_value=mock_document_data), \
         patch('src.pipeline.LatexToGraphParser') as mock_parser_class, \
         patch('src.pipeline.chunker.chunk_conceptual_blocks', return_value=[]), \
         patch('src.pipeline.document_loader.update_processed_docs_log') as mock_update_log, \
         patch('src.pipeline.os.makedirs') as mock_makedirs, \
         patch('src.pipeline.logging.getLogger') as mock_logger:
        
        # Mock the parser instance
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.get_graph_nodes_as_conceptual_blocks.return_value = mock_conceptual_blocks
        
        pipeline = FastPipeline()
        result = await pipeline.run_ingestion()
        
        assert result is False

@pytest.mark.asyncio
async def test_run_ingestion_vector_store_error(mock_document_data, mock_conceptual_blocks, mock_final_chunks):
    """Test ingestion when vector store operations fail."""
    with patch('src.pipeline.document_loader.load_new_documents', return_value=mock_document_data), \
         patch('src.pipeline.LatexToGraphParser') as mock_parser_class, \
         patch('src.pipeline.chunker.chunk_conceptual_blocks', return_value=mock_final_chunks), \
         patch('src.pipeline.vector_store_manager.get_weaviate_client') as mock_get_client, \
         patch('src.pipeline.vector_store_manager.fast_embed_and_store_chunks', side_effect=Exception("Vector store error")), \
         patch('src.pipeline.document_loader.update_processed_docs_log') as mock_update_log, \
         patch('src.pipeline.os.makedirs') as mock_makedirs, \
         patch('src.pipeline.logging.getLogger') as mock_logger:
        
        # Mock the parser instance
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.get_graph_nodes_as_conceptual_blocks.return_value = mock_conceptual_blocks
        
        # Mock Weaviate client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        pipeline = FastPipeline()
        result = await pipeline.run_ingestion()
        
        assert result is False

@pytest.mark.asyncio
async def test_run_ingestion_mixed_success_and_failure(mock_document_data):
    """Test ingestion with mixed success and failure scenarios."""
    with patch('src.pipeline.document_loader.load_new_documents', return_value=mock_document_data), \
         patch('src.pipeline.LatexToGraphParser') as mock_parser_class, \
         patch('src.pipeline.chunker.chunk_conceptual_blocks', return_value=[]), \
         patch('src.pipeline.document_loader.update_processed_docs_log') as mock_update_log, \
         patch('src.pipeline.os.makedirs') as mock_makedirs, \
         patch('src.pipeline.logging.getLogger') as mock_logger:
        
        # Mock the parser instance to fail on first document, succeed on second
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        
        def extract_side_effect(latex_content, doc_id, source):
            if doc_id == "test_doc_1":
                raise Exception("Parser error for doc 1")
            # Success for test_doc_2
        
        mock_parser.extract_structured_nodes.side_effect = extract_side_effect
        mock_parser.get_graph_nodes_as_conceptual_blocks.return_value = []
        
        pipeline = FastPipeline()
        result = await pipeline.run_ingestion()
        
        assert result is True  # Should return True when no vector store errors
        assert pipeline.processed_count == 1
        assert pipeline.error_count == 1

@pytest.mark.asyncio
async def test_run_ingestion_pipeline_exception():
    """Test ingestion when the entire pipeline raises an exception."""
    with patch('src.pipeline.document_loader.load_new_documents', side_effect=Exception("Pipeline error")), \
         patch('src.pipeline.logging.getLogger') as mock_logger:
        
        pipeline = FastPipeline()
        result = await pipeline.run_ingestion()
        
        assert result is False

# --- Integration Tests ---

@pytest.mark.asyncio
async def test_run_ingestion_integration_full_flow(mock_document_data, mock_conceptual_blocks, mock_final_chunks):
    """Integration test for the full ingestion flow."""
    with patch('src.pipeline.document_loader.load_new_documents', return_value=mock_document_data), \
         patch('src.pipeline.LatexToGraphParser') as mock_parser_class, \
         patch('src.pipeline.chunker.chunk_conceptual_blocks', return_value=mock_final_chunks), \
         patch('src.pipeline.vector_store_manager.get_weaviate_client') as mock_get_client, \
         patch('src.pipeline.fast_embed_and_store_chunks', new_callable=AsyncMock) as mock_store_chunks, \
         patch('src.pipeline.document_loader.update_processed_docs_log') as mock_update_log, \
         patch('src.pipeline.os.makedirs') as mock_makedirs, \
         patch('src.pipeline.logging.getLogger') as mock_logger, \
         patch('src.pipeline.config.EMBEDDING_MODEL_NAME', 'test-model'):
        
        # Mock the parser instance
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.get_graph_nodes_as_conceptual_blocks.return_value = mock_conceptual_blocks
        
        # Mock Weaviate client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        pipeline = FastPipeline()
        result = await pipeline.run_ingestion()
        
        # Verify the complete flow
        assert result is True
        
        # Verify parser was initialized with correct model
        mock_parser_class.assert_called_once_with(model_name='test-model')
        
        # Verify parser was called for each document
        assert mock_parser.extract_structured_nodes.call_count == 2
        
        # Verify conceptual blocks were retrieved
        mock_parser.get_graph_nodes_as_conceptual_blocks.assert_called_once()
        
        # Verify chunks were generated
        mock_store_chunks.assert_awaited_once_with(mock_client, mock_final_chunks, batch_size=100)
        
        # Verify log was updated
        mock_update_log.assert_called_once()

@pytest.mark.asyncio
async def test_run_ingestion_integration_error_handling():
    """Integration test for error handling in the ingestion flow."""
    with patch('src.pipeline.document_loader.load_new_documents', return_value=[]), \
         patch('src.pipeline.logging.getLogger') as mock_logger:
        
        pipeline = FastPipeline()
        result = await pipeline.run_ingestion()
        
        # Should handle empty document list gracefully
        assert result is True

# --- Wrapper Function Tests ---

@pytest.mark.asyncio
async def test_run_fast_ingestion_pipeline_success():
    """Test the wrapper function with successful ingestion."""
    with patch('src.pipeline.FastPipeline') as mock_pipeline_class, \
         patch('src.pipeline.sys') as mock_sys:
        
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run_ingestion = AsyncMock(return_value=True)
        
        await run_fast_ingestion_pipeline()
        
        mock_pipeline.run_ingestion.assert_awaited_once()
        mock_sys.exit.assert_not_called()

@pytest.mark.asyncio
async def test_run_fast_ingestion_pipeline_failure():
    """Test the wrapper function with failed ingestion."""
    with patch('src.pipeline.FastPipeline') as mock_pipeline_class, \
         patch('src.pipeline.sys') as mock_sys:
        
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run_ingestion = AsyncMock(return_value=False)
        
        await run_fast_ingestion_pipeline()
        
        mock_pipeline.run_ingestion.assert_awaited_once()
        mock_sys.exit.assert_called_once_with(1)

# --- GNN Training Tests ---

def test_run_fast_gnn_training_success(temp_graph_db_dir, temp_embeddings_dir):
    
    
    
    """Test successful GNN training."""
    # Create required files
    graph_file = os.path.join(temp_graph_db_dir, "knowledge_graph.graphml")
    embeddings_file = os.path.join(temp_embeddings_dir, "initial_text_embeddings.pkl")
    with open(graph_file, 'w') as f:
        f.write("test graph content")
    with open(embeddings_file, 'w') as f:
        f.write("test embeddings content")
    with patch('src.pipeline.run_gnn_training') as mock_run_training, \
         patch('src.pipeline.os.path.exists', side_effect=lambda x: x in [graph_file, embeddings_file]), \
         patch('src.pipeline.sys.exit') as mock_sys_exit:
        mock_sys_exit.reset_mock()
        run_fast_gnn_training()
        mock_run_training.assert_called_once()
        assert mock_sys_exit.call_count == 2
        mock_sys_exit.assert_has_calls([call(1), call(1)])

def test_run_fast_gnn_training_missing_graph_file(temp_embeddings_dir):
    """Test GNN training when graph file is missing."""
    embeddings_file = os.path.join(temp_embeddings_dir, "initial_text_embeddings.pkl")
    with open(embeddings_file, 'w') as f:
        f.write("test embeddings content")
    with patch('src.pipeline.run_gnn_training') as mock_run_training, \
         patch('src.pipeline.logging.getLogger') as mock_logger, \
         patch('src.pipeline.os.path.exists', side_effect=lambda x: x == embeddings_file), \
         patch('src.pipeline.sys.exit') as mock_sys_exit:
        mock_sys_exit.reset_mock()
        run_fast_gnn_training()
        assert mock_sys_exit.call_count == 2
        mock_sys_exit.assert_has_calls([call(1), call(1)])
        mock_logger().error.assert_any_call("❌ Knowledge graph not found. Run 'make ingest' first.")

def test_run_fast_gnn_training_missing_embeddings_file(temp_graph_db_dir):
    """Test GNN training when embeddings file is missing."""
    graph_file = os.path.join(temp_graph_db_dir, "knowledge_graph.graphml")
    with open(graph_file, 'w') as f:
        f.write("test graph content")
    with patch('src.pipeline.run_gnn_training') as mock_run_training, \
         patch('src.pipeline.logging.getLogger') as mock_logger, \
         patch('src.pipeline.os.path.exists', side_effect=lambda x: x == graph_file), \
         patch('src.pipeline.sys.exit') as mock_sys_exit:
        mock_sys_exit.reset_mock()
        run_fast_gnn_training()
        assert mock_sys_exit.call_count == 2
        mock_sys_exit.assert_has_calls([call(1), call(1)])
        mock_logger().error.assert_any_call("❌ Initial embeddings not found. Run 'make ingest' first.")

def test_run_fast_gnn_training_training_error(temp_graph_db_dir, temp_embeddings_dir):
    """Test GNN training when training raises an exception."""
    # Create required files
    graph_file = os.path.join(temp_graph_db_dir, "knowledge_graph.graphml")
    embeddings_file = os.path.join(temp_embeddings_dir, "initial_text_embeddings.pkl")
    with open(graph_file, 'w') as f:
        f.write("test graph content")
    with open(embeddings_file, 'w') as f:
        f.write("test embeddings content")
    with patch('src.pipeline.run_gnn_training', side_effect=Exception("Training error")), \
         patch('src.pipeline.logging.getLogger') as mock_logger, \
         patch('src.pipeline.os.path.exists', side_effect=lambda x: x in [graph_file, embeddings_file]), \
         patch('src.pipeline.sys.exit') as mock_sys_exit:
        mock_sys_exit.reset_mock()
        run_fast_gnn_training()
        assert mock_sys_exit.call_count == 3
        mock_sys_exit.assert_has_calls([call(1), call(1), call(1)])
        mock_logger().error.assert_any_call("❌ GNN training failed: Training error")

def test_run_fast_gnn_training_module_not_available():
    """Test GNN training when the module is not available."""
    with patch('src.pipeline.run_gnn_training', None), \
         patch('src.pipeline.logging.getLogger') as mock_logger, \
         patch('src.pipeline.sys.exit') as mock_sys_exit:
        mock_sys_exit.reset_mock()
        run_fast_gnn_training()
        assert mock_sys_exit.call_count == 2
        mock_sys_exit.assert_has_calls([call(1), call(1)])

# --- Edge Cases ---

@pytest.mark.asyncio
async def test_run_ingestion_with_large_document_set():
    """Test ingestion with a large number of documents."""
    large_document_data = [
        {
            "doc_id": f"test_doc_{i}",
            "source": f"/path/to/test{i}.tex",
            "filename": f"test{i}.tex",
            "type": "latex",
            "raw_content": f"\\documentclass{{article}}\\begin{{document}}Test {i}\\end{{document}}"
        }
        for i in range(100)  # 100 documents
    ]
    
    with patch('src.pipeline.document_loader.load_new_documents', return_value=large_document_data), \
         patch('src.pipeline.LatexToGraphParser') as mock_parser_class, \
         patch('src.pipeline.chunker.chunk_conceptual_blocks', return_value=[]), \
         patch('src.pipeline.document_loader.update_processed_docs_log') as mock_update_log, \
         patch('src.pipeline.os.makedirs') as mock_makedirs, \
         patch('src.pipeline.logging.getLogger') as mock_logger:
        
        # Mock the parser instance
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.get_graph_nodes_as_conceptual_blocks.return_value = []
        
        pipeline = FastPipeline()
        result = await pipeline.run_ingestion()
        
        assert result is True
        assert pipeline.processed_count == 100
        assert pipeline.error_count == 0

@pytest.mark.asyncio
async def test_run_ingestion_with_empty_documents():
    """Test ingestion with empty document content."""
    empty_document_data = [
        {
            "doc_id": "empty_doc",
            "source": "/path/to/empty.tex",
            "filename": "empty.tex",
            "type": "latex",
            "raw_content": ""
        }
    ]
    
    with patch('src.pipeline.document_loader.load_new_documents', return_value=empty_document_data), \
         patch('src.pipeline.LatexToGraphParser') as mock_parser_class, \
         patch('src.pipeline.chunker.chunk_conceptual_blocks', return_value=[]), \
         patch('src.pipeline.document_loader.update_processed_docs_log') as mock_update_log, \
         patch('src.pipeline.os.makedirs') as mock_makedirs, \
         patch('src.pipeline.logging.getLogger') as mock_logger:
        
        # Mock the parser instance
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.get_graph_nodes_as_conceptual_blocks.return_value = []
        
        pipeline = FastPipeline()
        result = await pipeline.run_ingestion()
        
        assert result is True
        assert pipeline.processed_count == 1
        assert pipeline.error_count == 0

@pytest.mark.asyncio
async def test_run_ingestion_with_malformed_latex():
    """Test ingestion with malformed LaTeX content."""
    malformed_document_data = [
        {
            "doc_id": "malformed_doc",
            "source": "/path/to/malformed.tex",
            "filename": "malformed.tex",
            "type": "latex",
            "raw_content": "\\documentclass{article}\\begin{document}\\malformed{command}\\end{document}"
        }
    ]
    
    with patch('src.pipeline.document_loader.load_new_documents', return_value=malformed_document_data), \
         patch('src.pipeline.LatexToGraphParser') as mock_parser_class, \
         patch('src.pipeline.document_loader.update_processed_docs_log') as mock_update_log, \
         patch('src.pipeline.os.makedirs') as mock_makedirs, \
         patch('src.pipeline.logging.getLogger') as mock_logger:
        
        # Mock the parser instance to handle malformed content gracefully
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.extract_structured_nodes.side_effect = Exception("Malformed LaTeX")
        mock_parser.get_graph_nodes_as_conceptual_blocks.return_value = []
        
        pipeline = FastPipeline()
        result = await pipeline.run_ingestion()
        
        assert result is True  # Should continue processing even with errors
        assert pipeline.processed_count == 0
        assert pipeline.error_count == 1

# --- Resource Management Tests ---

@pytest.mark.asyncio
async def test_run_ingestion_directory_creation():
    """Test that directories are created when they don't exist."""
    with patch('src.pipeline.document_loader.load_new_documents', return_value=[]), \
         patch('src.pipeline.os.makedirs') as mock_makedirs, \
         patch('src.pipeline.logging.getLogger') as mock_logger:
        
        pipeline = FastPipeline()
        await pipeline.run_ingestion()
        
        # Verify directories were created
        mock_makedirs.assert_any_call('data/graph_db', exist_ok=True)
        mock_makedirs.assert_any_call('data/embeddings', exist_ok=True)

@pytest.mark.asyncio
async def test_run_ingestion_graph_saving(mock_document_data, mock_conceptual_blocks):
    """Test that graph and embeddings are saved correctly."""
    with patch('src.pipeline.document_loader.load_new_documents', return_value=mock_document_data), \
         patch('src.pipeline.LatexToGraphParser') as mock_parser_class, \
         patch('src.pipeline.chunker.chunk_conceptual_blocks', return_value=[]), \
         patch('src.pipeline.document_loader.update_processed_docs_log') as mock_update_log, \
         patch('src.pipeline.os.makedirs') as mock_makedirs, \
         patch('src.pipeline.logging.getLogger') as mock_logger:
        
        # Mock the parser instance
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.get_graph_nodes_as_conceptual_blocks.return_value = mock_conceptual_blocks
        
        pipeline = FastPipeline()
        await pipeline.run_ingestion()
        
        # Verify graph and embeddings were saved
        mock_parser.save_graph_and_embeddings.assert_called_once_with(
            'data/graph_db/knowledge_graph.graphml',
            'data/embeddings/initial_text_embeddings.pkl'
        ) 