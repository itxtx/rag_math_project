import os
from .latex_parser import parse_latex_file
from .pdf_parser import parse_pdf_file
from src import config

def save_parsed_content(filename: str, content: str, output_dir: str):
    """Saves the parsed content to a file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    output_path = os.path.join(output_dir, f"{base_filename}_parsed.txt")
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved parsed content to: {output_path}")
    except Exception as e:
        print(f"Error saving parsed content for {filename} to {output_path}: {e}")


def load_and_parse_documents(pdf_math_tool: str = "general", process_pdfs=False):
    """
    Loads documents from specified directories, determines their type,
    calls the appropriate parser, and saves the parsed output.
    Returns a list of dictionaries, each containing source path, type, and parsed content.
    :param pdf_math_tool: Tool to use for math-heavy PDFs ('mathpix', 'nougat', 'general').
    """
    all_parsed_data = []

    # Process LaTeX files
    print(f"\n--- Processing LaTeX files from: {config.RAW_LATEX_DIR} ---")
    if not os.path.exists(config.RAW_LATEX_DIR):
        print(f"LaTeX directory not found: {config.RAW_LATEX_DIR}")
    else:
        for filename in os.listdir(config.RAW_LATEX_DIR):
            if filename.endswith(".tex"):
                file_path = os.path.join(config.RAW_LATEX_DIR, filename)
                print(f"Processing LaTeX file: {file_path}")
                content = parse_latex_file(file_path)
                if content:
                    all_parsed_data.append({"source": file_path, "type": "latex", "content": content})
                    save_parsed_content(filename, content, config.PARSED_LATEX_OUTPUT_DIR)
            else:
                print(f"Skipping non-LaTeX file in LaTeX directory: {filename}")

    # Process PDF files
    if process_pdfs == True:
        print(f"\n--- Processing PDF files from: {config.RAW_PDF_DIR} ---")
        if not os.path.exists(config.RAW_PDF_DIR):
            print(f"PDF directory not found: {config.RAW_PDF_DIR}")
        else:
            for filename in os.listdir(config.RAW_PDF_DIR):
                if filename.endswith(".pdf"):
                    file_path = os.path.join(config.RAW_PDF_DIR, filename)
                    print(f"Processing PDF file: {file_path}")
                    
                    # Heuristic to determine if PDF is math-heavy (can be improved)
                    is_math_heavy = "math" in filename.lower() or "paper" in filename.lower() or "notes" in filename.lower()
                    
                    # Use the specified tool for math-heavy PDFs, otherwise general
                    current_pdf_math_tool = pdf_math_tool if is_math_heavy else "general"

                    content = parse_pdf_file(file_path, is_math_heavy=is_math_heavy, math_extractor_tool=current_pdf_math_tool)
                    if content:
                        all_parsed_data.append({"source": file_path, "type": "pdf", "content": content})
                        save_parsed_content(filename, content, config.PARSED_PDF_OUTPUT_DIR)
                else:
                    print(f"Skipping non-PDF file in PDF directory: {filename}")
        
    return all_parsed_data

if __name__ == '__main__':
    # Create dummy files if they don't exist for testing
    # LaTeX dummy file creation is handled in latex_parser.py's __main__
    # PDF dummy file creation is handled in pdf_parser.py's __main__
    
    # To run this test effectively, ensure those dummy files are created by running:
    # python -m src.data_ingestion.latex_parser
    # python -m src.data_ingestion.pdf_parser
    # OR ensure you have sample files in data/raw_latex and data/raw_pdfs

    print("Running document loader (using 'general' for math PDFs by default for this test)...")
    # For a more specific test, you could pass pdf_math_tool="mathpix" or "nougat"
    # e.g., parsed_documents = load_and_parse_documents(pdf_math_tool="mathpix")
    parsed_documents = load_and_parse_documents() 
    
    print(f"\n--- Loaded and Parsed {len(parsed_documents)} documents in total ---")
    for doc_info in parsed_documents:
        print(f"Source: {doc_info['source']}, Type: {doc_info['type']}")
        # print(f"Content Start: {doc_info['content'][:100]}...") # Content can be long

