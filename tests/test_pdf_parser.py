import unittest
import os
import tempfile
import fitz # PyMuPDF
import requests
from unittest.mock import patch, MagicMock

from src.data_ingestion.pdf_parser import parse_pdf_file, general_pdf_extractor
from src import config # To access MATHPIX keys if needed

class TestPdfParser(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # Store original Mathpix keys and clear them for most tests
        self.original_mathpix_id = config.MATHPIX_APP_ID
        self.original_mathpix_key = config.MATHPIX_APP_KEY
        config.MATHPIX_APP_ID = None # Ensure Mathpix is not called unless specifically tested
        config.MATHPIX_APP_KEY = None

    def tearDown(self):
        # Restore original Mathpix keys
        config.MATHPIX_APP_ID = self.original_mathpix_id
        config.MATHPIX_APP_KEY = self.original_mathpix_key
        # Clean up temp files
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.test_dir)

    def create_temp_pdf_file(self, text_content, filename="test.pdf"):
        file_path = os.path.join(self.test_dir, filename)
        doc = fitz.open() # New empty PDF
        page = doc.new_page()
        # Simple text insertion. For complex layouts/math, this won't reflect real PDFs.
        page.insert_text((72, 72), text_content)
        doc.save(file_path)
        doc.close()
        return file_path

    def test_general_pdf_extractor_simple(self):
        expected_text = "This is a simple PDF for general extraction."
        file_path = self.create_temp_pdf_file(expected_text)
        extracted_text = general_pdf_extractor(file_path)
        self.assertIn(expected_text, extracted_text)

    def test_parse_pdf_file_general(self):
        expected_text = "General PDF content."
        file_path = self.create_temp_pdf_file(expected_text)
        # is_math_heavy = False, tool = "general" (default)
        extracted_text = parse_pdf_file(file_path, is_math_heavy=False)
        self.assertIn(expected_text, extracted_text)
    '''
    def test_parse_pdf_file_math_heavy_fallback_to_general(self):
        expected_text = "Math-heavy PDF, but no special tool configured."
        file_path = self.create_temp_pdf_file(expected_text)
        # is_math_heavy = True, but math_extractor_tool will default to general if not mathpix/nougat
        extracted_text = parse_pdf_file(file_path, is_math_heavy=True, math_extractor_tool="unsupported_tool")
        self.assertIn(expected_text, extracted_text, "Should fall back to general extractor")
        
        extracted_text_default_general = parse_pdf_file(file_path, is_math_heavy=True, math_extractor_tool="general")
        self.assertIn(expected_text, extracted_text_default_general, "Should use general extractor explicitly")


    def test_parse_pdf_file_nougat_placeholder(self):
        expected_text = "Nougat test PDF."
        file_path = self.create_temp_pdf_file(expected_text)
        # is_math_heavy = True, tool = "nougat" (currently a placeholder)
        with patch('src.data_ingestion.pdf_parser.math_pdf_extractor_nougat', return_value="Nougat processed (mocked)"):
            # Even if mocked, the current parse_pdf_file has a print and then calls general
            # So we test if it *tries* to call nougat path and falls back to general
            extracted_text = parse_pdf_file(file_path, is_math_heavy=True, math_extractor_tool="nougat")
        self.assertIn(expected_text, extracted_text, "Nougat is a placeholder, should fall back to general")


    @patch('src.data_ingestion.pdf_parser.requests.post')
    def test_parse_pdf_file_mathpix_success(self, mock_post):
        # Restore Mathpix keys for this specific test
        config.MATHPIX_APP_ID = "dummy_id"
        config.MATHPIX_APP_KEY = "dummy_key"

        expected_md_content = "Mathpix processed: $E=mc^2$"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"md": expected_md_content}
        mock_post.return_value = mock_response

        file_path = self.create_temp_pdf_file("PDF for Mathpix.")
        extracted_text = parse_pdf_file(file_path, is_math_heavy=True, math_extractor_tool="mathpix")
        
        mock_post.assert_called_once() # Check if API was called
        self.assertEqual(extracted_text, expected_md_content)

    @patch('src.data_ingestion.pdf_parser.requests.post')
    def test_parse_pdf_file_mathpix_api_error_fallback(self, mock_post):
        config.MATHPIX_APP_ID = "dummy_id"
        config.MATHPIX_APP_KEY = "dummy_key"

        mock_post.side_effect = requests.exceptions.RequestException("API Unreachable")
        
        fallback_text = "Mathpix API failed, fallback content."
        file_path = self.create_temp_pdf_file(fallback_text)
        
        extracted_text = parse_pdf_file(file_path, is_math_heavy=True, math_extractor_tool="mathpix")
        
        mock_post.assert_called_once()
        self.assertIn(fallback_text, extracted_text, "Should fall back to general extractor on Mathpix API error")

    def test_parse_pdf_file_mathpix_not_configured_fallback(self):
        # Ensure keys are None for this test
        config.MATHPIX_APP_ID = None
        config.MATHPIX_APP_KEY = None

        fallback_text = "Mathpix not configured, fallback content."
        file_path = self.create_temp_pdf_file(fallback_text)
        
        extracted_text = parse_pdf_file(file_path, is_math_heavy=True, math_extractor_tool="mathpix")
        self.assertIn(fallback_text, extracted_text, "Should fall back to general if Mathpix keys are not set")
    '''
    def test_empty_pdf_file(self):
        file_path = self.create_temp_pdf_file("") # Empty content
        extracted_text = parse_pdf_file(file_path)
        # PyMuPDF might return a newline or form feed for an empty page
        self.assertTrue(extracted_text.strip() == "" or extracted_text == "\f", "Empty PDF should result in empty or whitespace string")

    def test_pdf_file_not_found(self):
        extracted_text = parse_pdf_file("non_existent_file.pdf")
        self.assertEqual(extracted_text, "")


if __name__ == '__main__':
    unittest.main()