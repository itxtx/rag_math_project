import os
import fitz  # PyMuPDF
import requests # For API calls if using Mathpix
from src import config

def general_pdf_extractor(file_path: str) -> str:
    """
    Extracts text from a PDF using PyMuPDF.
    Suitable for PDFs without complex mathematical notation or where basic text is sufficient.
    """
    print(f"Using general PDF extractor for: {file_path}")
    try:
        doc = fitz.open(file_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text") # "text" gives plain text
        print(f"Successfully extracted text from {file_path} using PyMuPDF (general)")
        return text
    except Exception as e:
        print(f"Error in general_pdf_extractor for {file_path}: {e}")
        return ""

def math_pdf_extractor_mathpix_api(file_path: str) -> str:
    """
    Extracts content from a PDF, attempting to convert math to LaTeX using Mathpix API.
    Requires MATHPIX_APP_ID and MATHPIX_APP_KEY to be set in .env.
    """
    print(f"Attempting Mathpix API extraction for: {file_path}")
    if not config.MATHPIX_APP_ID or not config.MATHPIX_APP_KEY:
        print("Mathpix App ID or Key not configured. Skipping Mathpix extraction.")
        print("Falling back to general PDF extractor for this file.")
        return general_pdf_extractor(file_path)

    try:
        # Mathpix API endpoint for PDF processing
        # This is a simplified example. Refer to official Mathpix documentation for full options.
        # Common options include 'conversion_formats': {'md.latex': True} or {'tex.zip': True}
        # and 'math_inline_delimiters': ['$', '$'], 'math_display_delimiters': ['$$', '$$']
        # The response format will vary based on these options.
        # Here, we might aim for Markdown with LaTeX math.
        options_json = {
            "conversion_formats": {"md.latex": True}, # Get Markdown with LaTeX math
            "math_inline_delimiters": ["$", "$"],
            "math_display_delimiters": ["$$", "$$"],
            # "remove_section_numbering": True, # Example option
        }
        
        print("Sending PDF to Mathpix API. This may take some time...")
        with open(file_path, "rb") as f:
            response = requests.post(
                "https://api.mathpix.com/v3/pdf", # Check current API endpoint
                headers={
                    "app_id": config.MATHPIX_APP_ID,
                    "app_key": config.MATHPIX_APP_KEY,
                },
                files={"file": f},
                data={"options_json": str(options_json)} # Send options as a string
            )
        
        response.raise_for_status() # Raise an exception for bad status codes
        
        result = response.json()
        if "md" in result: # If we requested Markdown output
            print(f"Successfully extracted Markdown with LaTeX from {file_path} using Mathpix API.")
            return result["md"]
        elif "text" in result: # Fallback if 'md' isn't there but 'text' is
             print(f"Mathpix API returned text for {file_path}.")
             return result["text"]
        elif "error" in result:
            print(f"Mathpix API error for {file_path}: {result.get('error_info', {}).get('message', result['error'])}")
            return ""
        else:
            print(f"Unexpected Mathpix API response structure for {file_path}. Result: {result}")
            return ""

    except requests.exceptions.RequestException as e:
        print(f"Mathpix API request failed for {file_path}: {e}")
    except Exception as e:
        print(f"Error in math_pdf_extractor_mathpix_api for {file_path}: {e}")
    
    print("Mathpix extraction failed or not configured. Falling back to general PDF extractor.")
    return general_pdf_extractor(file_path) # Fallback

def math_pdf_extractor_nougat(file_path: str) -> str:
    """
    Placeholder for extracting content using a local Nougat model.
    Actual implementation would involve loading the Nougat model and processing the PDF.
    """
    print(f"Using Nougat (placeholder) for math PDF extraction: {file_path}")
    # TODO: User needs to implement Nougat integration here.
    # This would involve:
    # 1. Setting up Nougat (https://github.com/facebookresearch/nougat)
    # 2. Calling Nougat's processing functions on the PDF file.
    # 3. Nougat typically outputs a .mmd (Mathpix Markdown) file. Read that content.
    print("Nougat extractor is a placeholder. Implement Nougat processing logic.")
    print("Falling back to general PDF extractor for this file.")
    return general_pdf_extractor(file_path)


def parse_pdf_file(file_path: str, is_math_heavy: bool = False, math_extractor_tool: str = "general") -> str:
    """
    Parses a .pdf file. Routes to specialized math extractor if specified.
    :param file_path: Path to the PDF file.
    :param is_math_heavy: Boolean flag indicating if the PDF is math-heavy.
    :param math_extractor_tool: Specifies the tool for math extraction ('mathpix', 'nougat', 'general').
    """
    print(f"Parsing PDF file: {file_path} (Math heavy: {is_math_heavy}, Tool: {math_extractor_tool})")
    
    if is_math_heavy:
        if math_extractor_tool == "mathpix" and config.MATHPIX_APP_ID and config.MATHPIX_APP_KEY:
            return math_pdf_extractor_mathpix_api(file_path)
        elif math_extractor_tool == "nougat":
            # return math_pdf_extractor_nougat(file_path) # Placeholder
            print("Nougat selected, but it's a placeholder. Using general extractor.")
            return general_pdf_extractor(file_path)
        else:
            if is_math_heavy and math_extractor_tool != "general":
                 print(f"Math-heavy PDF specified with tool '{math_extractor_tool}', but it's not configured or supported. Using general extractor.")
            return general_pdf_extractor(file_path)
    else:
        return general_pdf_extractor(file_path)

if __name__ == '__main__':
    # Example usage
    # Ensure you have a sample PDF in data/raw_pdfs/
    # e.g., convert the LaTeX sample to PDF and name it 'parser_test_sample.pdf'
    # and perhaps another one named 'parser_test_sample_math.pdf'
    
    dummy_pdf_filename = "parser_test_sample.pdf"
    dummy_math_pdf_filename = "parser_test_sample_math.pdf" # For testing math extraction

    dummy_pdf_file = os.path.join(config.RAW_PDF_DIR, dummy_pdf_filename)
    dummy_math_pdf_file = os.path.join(config.RAW_PDF_DIR, dummy_math_pdf_filename)

    if not os.path.exists(config.RAW_PDF_DIR):
        os.makedirs(config.RAW_PDF_DIR)

    # Create a dummy general PDF if it doesn't exist (e.g., from simple text)
    if not os.path.exists(dummy_pdf_file):
        try:
            doc = fitz.open() # new empty PDF
            page = doc.new_page()
            page.insert_text((72, 72), "This is a general test PDF content.\nNo complex math here.")
            doc.save(dummy_pdf_file)
            print(f"Created dummy PDF: {dummy_pdf_file}")
        except Exception as e:
            print(f"Could not create dummy PDF {dummy_pdf_file}: {e}")

    # Create a dummy math PDF (similar content for now, as math extraction is complex)
    if not os.path.exists(dummy_math_pdf_file):
        try:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "This is a math test PDF.\nConsider formula $E=mc^2$ and also display: $$ \\sum x_i $$")
            doc.save(dummy_math_pdf_file)
            print(f"Created dummy PDF: {dummy_math_pdf_file}")
        except Exception as e:
            print(f"Could not create dummy PDF {dummy_math_pdf_file}: {e}")


    if os.path.exists(dummy_pdf_file):
        print(f"\n--- Parsing General PDF: {dummy_pdf_filename} ---")
        parsed_content_general = parse_pdf_file(dummy_pdf_file, is_math_heavy=False)
        print("\nParsed General PDF Content (First 500 chars):\n", parsed_content_general[:500])
        
        output_filename_general = os.path.splitext(dummy_pdf_filename)[0] + "_parsed.txt"
        output_path_general = os.path.join(config.PARSED_PDF_OUTPUT_DIR, output_filename_general)
        with open(output_path_general, "w", encoding="utf-8") as f_out:
            f_out.write(parsed_content_general)
        print(f"Saved parsed general PDF content to: {output_path_general}")

    if os.path.exists(dummy_math_pdf_file):
        print(f"\n--- Parsing Math PDF (attempting Mathpix if configured, else fallback): {dummy_math_pdf_filename} ---")
        # To test Mathpix, set MATHPIX_APP_ID and MATHPIX_APP_KEY in your .env file
        parsed_content_math = parse_pdf_file(dummy_math_pdf_file, is_math_heavy=True, math_extractor_tool="mathpix")
        print("\nParsed Math PDF Content (First 500 chars):\n", parsed_content_math[:500])

        output_filename_math = os.path.splitext(dummy_math_pdf_filename)[0] + "_parsed.txt"
        output_path_math = os.path.join(config.PARSED_PDF_OUTPUT_DIR, output_filename_math)
        with open(output_path_math, "w", encoding="utf-8") as f_out:
            f_out.write(parsed_content_math)
        print(f"Saved parsed math PDF content to: {output_path_math}")
        
        # print(f"\n--- Parsing Math PDF (attempting Nougat - placeholder): {dummy_math_pdf_filename} ---")
        # parsed_content_nougat = parse_pdf_file(dummy_math_pdf_file, is_math_heavy=True, math_extractor_tool="nougat")
        # print("\nParsed Math PDF Content (Nougat - Placeholder):\n", parsed_content_nougat[:500])
