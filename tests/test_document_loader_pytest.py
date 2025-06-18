import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

from src.data_ingestion import document_loader
from src import config

# --- Fixtures ---

@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def temp_latex_dir(temp_data_dir):
    """Create a temporary LaTeX directory."""
    latex_dir = os.path.join(temp_data_dir, "latex")
    os.makedirs(latex_dir, exist_ok=True)
    return latex_dir

@pytest.fixture
def temp_pdf_dir(temp_data_dir):
    """Create a temporary PDF directory."""
    pdf_dir = os.path.join(temp_data_dir, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    return pdf_dir

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
def sample_pdf_content():
    """Sample PDF content (binary) for testing."""
    # This is a minimal PDF file content
    return b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF'

@pytest.fixture
def temp_log_file():
    """Create a temporary log file for testing."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log')
    temp_file.close()
    yield temp_file.name
    os.unlink(temp_file.name)

# --- Load Processed Doc Filenames Tests ---

def test_load_processed_doc_filenames_existing_file(temp_log_file):
    """Test loading processed document filenames from an existing log file."""
    # Create a log file with some filenames
    test_filenames = ["document1.tex", "document2.tex", "document3.pdf"]
    with open(temp_log_file, 'w', encoding='utf-8') as f:
        for filename in test_filenames:
            f.write(f"{filename}\n")
    
    result = document_loader.load_processed_doc_filenames(temp_log_file)
    
    assert isinstance(result, set)
    assert len(result) == 3
    assert "document1.tex" in result
    assert "document2.tex" in result
    assert "document3.pdf" in result

def test_load_processed_doc_filenames_nonexistent_file():
    """Test loading processed document filenames from a non-existent file."""
    nonexistent_file = "/nonexistent/path/to/file.log"
    
    result = document_loader.load_processed_doc_filenames(nonexistent_file)
    
    assert isinstance(result, set)
    assert len(result) == 0

def test_load_processed_doc_filenames_empty_file(temp_log_file):
    """Test loading processed document filenames from an empty file."""
    # Create an empty log file
    with open(temp_log_file, 'w', encoding='utf-8') as f:
        pass
    
    result = document_loader.load_processed_doc_filenames(temp_log_file)
    
    assert isinstance(result, set)
    assert len(result) == 0

def test_load_processed_doc_filenames_with_whitespace(temp_log_file):
    """Test loading processed document filenames with whitespace."""
    # Create a log file with filenames that have whitespace
    test_filenames = ["  document1.tex  ", "\ndocument2.tex\n", "document3.pdf"]
    with open(temp_log_file, 'w', encoding='utf-8') as f:
        for filename in test_filenames:
            f.write(f"{filename}\n")
    
    result = document_loader.load_processed_doc_filenames(temp_log_file)
    
    assert isinstance(result, set)
    # The function strips whitespace, so we expect 3 unique filenames
    assert len(result) == 3
    assert "document1.tex" in result  # Whitespace should be stripped
    assert "document2.tex" in result  # Newlines should be stripped
    assert "document3.pdf" in result

def test_load_processed_doc_filenames_duplicate_entries(temp_log_file):
    """Test loading processed document filenames with duplicate entries."""
    # Create a log file with duplicate filenames
    test_filenames = ["document1.tex", "document2.tex", "document1.tex", "document3.pdf"]
    with open(temp_log_file, 'w', encoding='utf-8') as f:
        for filename in test_filenames:
            f.write(f"{filename}\n")
    
    result = document_loader.load_processed_doc_filenames(temp_log_file)
    
    assert isinstance(result, set)
    assert len(result) == 3  # Duplicates are automatically removed by set
    assert "document1.tex" in result
    assert "document2.tex" in result
    assert "document3.pdf" in result

# --- Update Processed Docs Log Tests ---

def test_update_processed_docs_log_new_entries(temp_log_file):
    """Test updating processed docs log with new entries."""
    new_filenames = ["new_doc1.tex", "new_doc2.pdf", "new_doc3.tex"]
    
    document_loader.update_processed_docs_log(temp_log_file, new_filenames)
    
    # Verify the entries were added
    with open(temp_log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for filename in new_filenames:
        assert filename in content

def test_update_processed_docs_log_empty_list(temp_log_file):
    """Test updating processed docs log with empty list."""
    # Create initial content
    with open(temp_log_file, 'w', encoding='utf-8') as f:
        f.write("existing_doc.tex\n")
    
    initial_content = ""
    with open(temp_log_file, 'r', encoding='utf-8') as f:
        initial_content = f.read()
    
    document_loader.update_processed_docs_log(temp_log_file, [])
    
    # Verify content hasn't changed
    with open(temp_log_file, 'r', encoding='utf-8') as f:
        final_content = f.read()
    
    assert final_content == initial_content

def test_update_processed_docs_log_nonexistent_directory():
    """Test updating processed docs log when directory doesn't exist."""
    # Use a temporary directory that we can write to
    temp_dir = tempfile.mkdtemp()
    try:
        nonexistent_log_file = os.path.join(temp_dir, "nonexistent", "directory", "file.log")
        new_filenames = ["test_doc.tex"]
        
        # Should not raise an exception, should create the directory
        document_loader.update_processed_docs_log(nonexistent_log_file, new_filenames)
        
        # Verify the file was created
        assert os.path.exists(nonexistent_log_file)
        
        # Verify content was written
        with open(nonexistent_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "test_doc.tex" in content
    finally:
        shutil.rmtree(temp_dir)

def test_update_processed_docs_log_append_mode(temp_log_file):
    """Test that update_processed_docs_log appends to existing content."""
    # Create initial content
    initial_filenames = ["existing_doc1.tex", "existing_doc2.pdf"]
    with open(temp_log_file, 'w', encoding='utf-8') as f:
        for filename in initial_filenames:
            f.write(f"{filename}\n")
    
    # Add new filenames
    new_filenames = ["new_doc1.tex", "new_doc2.pdf"]
    document_loader.update_processed_docs_log(temp_log_file, new_filenames)
    
    # Verify both old and new content are present
    with open(temp_log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for filename in initial_filenames + new_filenames:
        assert filename in content

# --- Load New Documents Tests ---

@patch('src.data_ingestion.document_loader.config')
def test_load_new_documents_no_new_files(mock_config, temp_latex_dir, temp_log_file):
    """Test loading new documents when no new files exist."""
    mock_config.DATA_DIR_RAW_LATEX = temp_latex_dir
    mock_config.PROCESSED_DOCS_LOG_FILE = temp_log_file
    
    # Create a log file with all existing files
    existing_files = ["doc1.tex", "doc2.tex"]
    with open(temp_log_file, 'w', encoding='utf-8') as f:
        for filename in existing_files:
            f.write(f"{filename}\n")
    
    # Create the files in the latex directory
    for filename in existing_files:
        with open(os.path.join(temp_latex_dir, filename), 'w', encoding='utf-8') as f:
            f.write("\\documentclass{article}\\begin{document}Test\\end{document}")
    
    result = document_loader.load_new_documents()
    
    assert isinstance(result, list)
    assert len(result) == 0

@patch('src.data_ingestion.document_loader.config')
def test_load_new_documents_new_latex_files(mock_config, temp_latex_dir, temp_log_file, sample_latex_content):
    """Test loading new LaTeX documents."""
    mock_config.DATA_DIR_RAW_LATEX = temp_latex_dir
    mock_config.PROCESSED_DOCS_LOG_FILE = temp_log_file
    
    # Create a log file with some existing files
    existing_files = ["existing_doc.tex"]
    with open(temp_log_file, 'w', encoding='utf-8') as f:
        for filename in existing_files:
            f.write(f"{filename}\n")
    
    # Create existing file
    with open(os.path.join(temp_latex_dir, "existing_doc.tex"), 'w', encoding='utf-8') as f:
        f.write("\\documentclass{article}\\begin{document}Existing\\end{document}")
    
    # Create new files
    new_files = ["new_doc1.tex", "new_doc2.tex"]
    for filename in new_files:
        with open(os.path.join(temp_latex_dir, filename), 'w', encoding='utf-8') as f:
            f.write(sample_latex_content)
    
    result = document_loader.load_new_documents()
    
    assert isinstance(result, list)
    assert len(result) == 2
    
    # Check that only new files are included
    result_filenames = [doc['filename'] for doc in result]
    assert "new_doc1.tex" in result_filenames
    assert "new_doc2.tex" in result_filenames
    assert "existing_doc.tex" not in result_filenames
    
    # Check document structure
    for doc in result:
        assert 'doc_id' in doc
        assert 'source' in doc
        assert 'filename' in doc
        assert 'type' in doc
        assert 'raw_content' in doc
        assert doc['type'] == 'latex'
        assert doc['raw_content'] == sample_latex_content

@patch('src.data_ingestion.document_loader.config')
def test_load_new_documents_ignores_hidden_files(mock_config, temp_latex_dir, temp_log_file):
    """Test that hidden files are ignored."""
    mock_config.DATA_DIR_RAW_LATEX = temp_latex_dir
    mock_config.PROCESSED_DOCS_LOG_FILE = temp_log_file
    
    # Create hidden files
    hidden_files = [".hidden.tex", ".DS_Store", "._file.tex"]
    for filename in hidden_files:
        with open(os.path.join(temp_latex_dir, filename), 'w', encoding='utf-8') as f:
            f.write("\\documentclass{article}\\begin{document}Hidden\\end{document}")
    
    # Create visible file
    with open(os.path.join(temp_latex_dir, "visible.tex"), 'w', encoding='utf-8') as f:
        f.write("\\documentclass{article}\\begin{document}Visible\\end{document}")
    
    result = document_loader.load_new_documents()
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]['filename'] == "visible.tex"

@patch('src.data_ingestion.document_loader.config')
def test_load_new_documents_ignores_non_tex_files(mock_config, temp_latex_dir, temp_log_file):
    """Test that non-LaTeX files are ignored."""
    mock_config.DATA_DIR_RAW_LATEX = temp_latex_dir
    mock_config.PROCESSED_DOCS_LOG_FILE = temp_log_file
    
    # Create non-LaTeX files
    non_tex_files = ["document.txt", "document.md", "document.pdf", "document.tex.bak"]
    for filename in non_tex_files:
        with open(os.path.join(temp_latex_dir, filename), 'w', encoding='utf-8') as f:
            f.write("Content")
    
    # Create LaTeX file
    with open(os.path.join(temp_latex_dir, "document.tex"), 'w', encoding='utf-8') as f:
        f.write("\\documentclass{article}\\begin{document}LaTeX\\end{document}")
    
    result = document_loader.load_new_documents()
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]['filename'] == "document.tex"

@patch('src.data_ingestion.document_loader.config')
def test_load_new_documents_nonexistent_directory(mock_config, temp_log_file):
    """Test behavior when LaTeX directory doesn't exist."""
    mock_config.DATA_DIR_RAW_LATEX = "/nonexistent/latex/directory"
    mock_config.PROCESSED_DOCS_LOG_FILE = temp_log_file
    
    result = document_loader.load_new_documents()
    
    assert isinstance(result, list)
    assert len(result) == 0

@patch('src.data_ingestion.document_loader.config')
def test_load_new_documents_file_read_error(mock_config, temp_latex_dir, temp_log_file):
    """Test handling of file read errors."""
    mock_config.DATA_DIR_RAW_LATEX = temp_latex_dir
    mock_config.PROCESSED_DOCS_LOG_FILE = temp_log_file
    
    # Create a file that will cause a read error (directory instead of file)
    problematic_file = os.path.join(temp_latex_dir, "problematic.tex")
    os.makedirs(problematic_file, exist_ok=True)
    
    # Create a valid file
    with open(os.path.join(temp_latex_dir, "valid.tex"), 'w', encoding='utf-8') as f:
        f.write("\\documentclass{article}\\begin{document}Valid\\end{document}")
    
    result = document_loader.load_new_documents()
    
    # Should still process valid files even if one fails
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]['filename'] == "valid.tex"

@patch('src.data_ingestion.document_loader.config')
def test_load_new_documents_encoding_error(mock_config, temp_latex_dir, temp_log_file):
    """Test handling of encoding errors."""
    mock_config.DATA_DIR_RAW_LATEX = temp_latex_dir
    mock_config.PROCESSED_DOCS_LOG_FILE = temp_log_file
    
    # Create a file with invalid encoding
    with open(os.path.join(temp_latex_dir, "invalid_encoding.tex"), 'wb') as f:
        f.write(b'\xff\xfe\x00\x00')  # Invalid UTF-8
    
    # Create a valid file
    with open(os.path.join(temp_latex_dir, "valid.tex"), 'w', encoding='utf-8') as f:
        f.write("\\documentclass{article}\\begin{document}Valid\\end{document}")
    
    result = document_loader.load_new_documents()
    
    # Should still process valid files even if one has encoding issues
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]['filename'] == "valid.tex"

@patch('src.data_ingestion.document_loader.config')
def test_load_new_documents_empty_file(mock_config, temp_latex_dir, temp_log_file):
    """Test handling of empty files."""
    mock_config.DATA_DIR_RAW_LATEX = temp_latex_dir
    mock_config.PROCESSED_DOCS_LOG_FILE = temp_log_file
    
    # Create an empty file
    with open(os.path.join(temp_latex_dir, "empty.tex"), 'w', encoding='utf-8') as f:
        pass
    
    result = document_loader.load_new_documents()
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]['filename'] == "empty.tex"
    assert result[0]['raw_content'] == ""

@patch('src.data_ingestion.document_loader.config')
def test_load_new_documents_large_file(mock_config, temp_latex_dir, temp_log_file):
    """Test handling of large files."""
    mock_config.DATA_DIR_RAW_LATEX = temp_latex_dir
    mock_config.PROCESSED_DOCS_LOG_FILE = temp_log_file
    
    # Create a large file (1MB of content)
    large_content = "\\documentclass{article}\\begin{document}\\n" + "x" * 1000000 + "\\end{document}"
    
    with open(os.path.join(temp_latex_dir, "large.tex"), 'w', encoding='utf-8') as f:
        f.write(large_content)
    
    result = document_loader.load_new_documents()
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]['filename'] == "large.tex"
    assert len(result[0]['raw_content']) == len(large_content)

@patch('src.data_ingestion.document_loader.config')
def test_load_new_documents_doc_id_generation(mock_config, temp_latex_dir, temp_log_file):
    """Test that doc_id is correctly generated from filename."""
    mock_config.DATA_DIR_RAW_LATEX = temp_latex_dir
    mock_config.PROCESSED_DOCS_LOG_FILE = temp_log_file
    
    # Create files with different naming patterns
    test_files = [
        "simple.tex",
        "complex_name_with_underscores.tex",
        "file.with.dots.tex",
        "UPPERCASE.tex"
    ]
    
    for filename in test_files:
        with open(os.path.join(temp_latex_dir, filename), 'w', encoding='utf-8') as f:
            f.write("\\documentclass{article}\\begin{document}Test\\end{document}")
    
    result = document_loader.load_new_documents()
    
    assert len(result) == 4
    
    # Check doc_id generation
    doc_ids = [doc['doc_id'] for doc in result]
    assert "simple" in doc_ids
    assert "complex_name_with_underscores" in doc_ids
    assert "file.with.dots" in doc_ids
    assert "UPPERCASE" in doc_ids

# --- Edge Cases ---

@patch('src.data_ingestion.document_loader.config')
def test_load_new_documents_special_characters_in_filename(mock_config, temp_latex_dir, temp_log_file):
    """Test handling of special characters in filenames."""
    mock_config.DATA_DIR_RAW_LATEX = temp_latex_dir
    mock_config.PROCESSED_DOCS_LOG_FILE = temp_log_file
    
    # Create file with special characters
    special_filename = "document_with_special_chars_éñ_🧪.tex"
    with open(os.path.join(temp_latex_dir, special_filename), 'w', encoding='utf-8') as f:
        f.write("\\documentclass{article}\\begin{document}Special\\end{document}")
    
    result = document_loader.load_new_documents()
    
    assert len(result) == 1
    assert result[0]['filename'] == special_filename
    assert result[0]['doc_id'] == "document_with_special_chars_éñ_🧪"

@patch('src.data_ingestion.document_loader.config')
def test_load_new_documents_concurrent_access(mock_config, temp_latex_dir, temp_log_file):
    """Test behavior under concurrent access scenarios."""
    mock_config.DATA_DIR_RAW_LATEX = temp_latex_dir
    mock_config.PROCESSED_DOCS_LOG_FILE = temp_log_file
    
    # Create a file
    with open(os.path.join(temp_latex_dir, "test.tex"), 'w', encoding='utf-8') as f:
        f.write("\\documentclass{article}\\begin{document}Test\\end{document}")
    
    # Simulate concurrent access by reading the file multiple times
    results = []
    for _ in range(5):
        result = document_loader.load_new_documents()
        results.append(result)
    
    # All results should be the same
    for result in results:
        assert len(result) == 1
        assert result[0]['filename'] == "test.tex"

@patch('src.data_ingestion.document_loader.config')
def test_load_new_documents_permission_error(mock_config, temp_latex_dir, temp_log_file):
    """Test handling of permission errors."""
    mock_config.DATA_DIR_RAW_LATEX = temp_latex_dir
    mock_config.PROCESSED_DOCS_LOG_FILE = temp_log_file
    
    # Create a file with no read permissions
    restricted_file = os.path.join(temp_latex_dir, "restricted.tex")
    with open(restricted_file, 'w', encoding='utf-8') as f:
        f.write("\\documentclass{article}\\begin{document}Restricted\\end{document}")
    
    # Remove read permissions
    os.chmod(restricted_file, 0o000)
    
    # Create a valid file
    with open(os.path.join(temp_latex_dir, "valid.tex"), 'w', encoding='utf-8') as f:
        f.write("\\documentclass{article}\\begin{document}Valid\\end{document}")
    
    result = document_loader.load_new_documents()
    
    # Should still process valid files even if one has permission issues
    assert len(result) == 1
    assert result[0]['filename'] == "valid.tex"
    
    # Restore permissions for cleanup
    os.chmod(restricted_file, 0o644) 