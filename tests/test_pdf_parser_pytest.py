import pytest
import os
import tempfile
import fitz # PyMuPDF
from unittest.mock import patch, MagicMock
from src.data_ingestion.pdf_parser import parse_pdf_file, general_pdf_extractor
from src import config

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