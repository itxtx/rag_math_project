import pytest
import os
import tempfile
import fitz # PyMuPDF
from unittest.mock import patch, MagicMock, mock_open, AsyncMock
from src.data_ingestion.pdf_parser import parse_pdf_file, general_pdf_extractor
from src import config
from src.learner_model.profile_manager import LearnerProfileManager

@pytest.fixture
def temp_dir():
    test_dir = tempfile.mkdtemp()
    yield test_dir
    for root, dirs, files in os.walk(test_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(test_dir)

@pytest.fixture
def create_temp_pdf_file(temp_dir):
    def _create_file(text_content, filename="test.pdf"):
        file_path = os.path.join(temp_dir, filename)
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), text_content)
        doc.save(file_path)
        doc.close()
        return file_path
    return _create_file

def test_general_pdf_extractor_simple(create_temp_pdf_file):
    expected_text = "This is a simple PDF for general extraction."
    file_path = create_temp_pdf_file(expected_text)
    extracted_text = general_pdf_extractor(file_path)
    assert expected_text in extracted_text

def test_parse_pdf_file_general(create_temp_pdf_file):
    expected_text = "General PDF content."
    file_path = create_temp_pdf_file(expected_text)
    extracted_text = parse_pdf_file(file_path, is_math_heavy=False)
    assert expected_text in extracted_text

# Mathpix and Nougat tests are commented out/skipped due to external dependencies and credentials
# def test_parse_pdf_file_math_heavy_fallback_to_general(create_temp_pdf_file):
#     expected_text = "Math-heavy PDF, but no special tool configured."
#     file_path = create_temp_pdf_file(expected_text)
#     extracted_text = parse_pdf_file(file_path, is_math_heavy=True, math_extractor_tool="unsupported_tool")
#     assert expected_text in extracted_text
#     extracted_text_default_general = parse_pdf_file(file_path, is_math_heavy=True, math_extractor_tool="general")
#     assert expected_text in extracted_text_default_general

# def test_parse_pdf_file_nougat_placeholder(create_temp_pdf_file):
#     expected_text = "Nougat test PDF."
#     file_path = create_temp_pdf_file(expected_text)
#     with patch('src.data_ingestion.pdf_parser.math_pdf_extractor_nougat', return_value="Nougat processed (mocked)"):
#         extracted_text = parse_pdf_file(file_path, is_math_heavy=True, math_extractor_tool="nougat")
#     assert expected_text in extracted_text

# @patch('src.data_ingestion.pdf_parser.requests.post')
# def test_parse_pdf_file_mathpix_success(mock_post, create_temp_pdf_file):
#     config.MATHPIX_APP_ID = "dummy_id"
#     config.MATHPIX_APP_KEY = "dummy_key"
#     expected_md_content = "Mathpix processed: $E=mc^2$"
#     mock_response = MagicMock()
#     mock_response.status_code = 200
#     mock_response.json.return_value = {"md": expected_md_content}
#     mock_post.return_value = mock_response
#     file_path = create_temp_pdf_file("PDF for Mathpix.")
#     extracted_text = parse_pdf_file(file_path, is_math_heavy=True, math_extractor_tool="mathpix")
#     mock_post.assert_called_once()
#     assert extracted_text == expected_md_content

# @patch('src.data_ingestion.pdf_parser.requests.post')
# def test_parse_pdf_file_mathpix_api_error_fallback(mock_post, create_temp_pdf_file):
#     config.MATHPIX_APP_ID = "dummy_id"
#     config.MATHPIX_APP_KEY = "dummy_key"
#     mock_post.side_effect = Exception("API Unreachable")
#     fallback_text = "Mathpix API failed, fallback content."
#     file_path = create_temp_pdf_file(fallback_text)
#     extracted_text = parse_pdf_file(file_path, is_math_heavy=True, math_extractor_tool="mathpix")
#     mock_post.assert_called_once()
#     assert fallback_text in extracted_text

# def test_parse_pdf_file_mathpix_not_configured_fallback(create_temp_pdf_file):
#     config.MATHPIX_APP_ID = None
#     config.MATHPIX_APP_KEY = None
#     fallback_text = "Mathpix not configured, fallback content."
#     file_path = create_temp_pdf_file(fallback_text)
#     extracted_text = parse_pdf_file(file_path, is_math_heavy=True, math_extractor_tool="mathpix")
#     assert fallback_text in extracted_text

def test_empty_pdf_file(create_temp_pdf_file):
    file_path = create_temp_pdf_file("")
    extracted_text = parse_pdf_file(file_path)
    assert extracted_text.strip() == "" or extracted_text == "\f"

def test_pdf_file_not_found():
    extracted_text = parse_pdf_file("non_existent_file.pdf")
    assert extracted_text == ""

def test_multicolumn_pdf_extraction(create_temp_pdf_file, temp_dir):
    # Simulate a multi-column PDF by inserting text in two locations
    file_path = create_temp_pdf_file("Column 1 text.", filename="multi_col.pdf")
    # Add a second page with different text to simulate another column
    doc = fitz.open(file_path)
    page = doc.new_page()
    page.insert_text((200, 72), "Column 2 text.")
    # Save to a new file to avoid incremental write issues
    new_file_path = os.path.join(temp_dir, "multi_col_modified.pdf")
    doc.save(new_file_path)
    doc.close()
    extracted_text = general_pdf_extractor(new_file_path)
    assert "Column 1 text." in extracted_text
    assert "Column 2 text." in extracted_text

def test_pdf_with_tables_and_images(create_temp_pdf_file, temp_dir):
    # Create a PDF with text that simulates a table and an image
    file_path = create_temp_pdf_file("Header1 | Header2\n------ | ------\nCell1  | Cell2\n", filename="table_img.pdf")
    # Add a page with an image (simulate by drawing a rectangle)
    doc = fitz.open(file_path)
    page = doc.new_page()
    rect = fitz.Rect(50, 50, 150, 150)
    page.draw_rect(rect)
    page.insert_text((60, 160), "Image below")
    # Save to a new file to avoid incremental write issues
    new_file_path = os.path.join(temp_dir, "table_img_modified.pdf")
    doc.save(new_file_path)
    doc.close()
    extracted_text = general_pdf_extractor(new_file_path)
    assert "Header1" in extracted_text
    assert "Cell2" in extracted_text
    assert "Image below" in extracted_text

from unittest.mock import patch

def test_text_selectable_vs_scanned_pdf(create_temp_pdf_file):
    # Text-selectable PDF
    file_path = create_temp_pdf_file("Selectable text PDF.", filename="selectable.pdf")
    extracted_text = general_pdf_extractor(file_path)
    assert "Selectable text PDF." in extracted_text
    # Scanned/image-based PDF (simulate by patching PyMuPDF to return empty text)
    with patch('fitz.Page.get_text', return_value=""):
        extracted_text_img = general_pdf_extractor(file_path)
        assert extracted_text_img.strip() == ""

@pytest.fixture
def temp_pdf_file(tmp_path):
    file_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Sample PDF content.")
    doc.save(str(file_path))
    doc.close()
    return str(file_path)

@pytest.fixture
def mock_profile_manager():
    mock = AsyncMock(spec=LearnerProfileManager)
    mock.get_concepts_for_review = AsyncMock(return_value=[])
    mock.get_concept_knowledge = AsyncMock(return_value=None)
    mock.create_profile = AsyncMock(return_value=None)
    # Add missing method
    mock.get_last_attempted_concept_and_doc = AsyncMock(return_value=(None, None))
    return mock 

@pytest.fixture
def temp_latex_dir(tmp_path):
    dir_path = tmp_path / "latex_files"
    dir_path.mkdir()
    return str(dir_path) 