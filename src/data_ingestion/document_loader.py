import os
from .latex_parser import parse_latex_file
from .pdf_parser import parse_pdf_file
from src import config # To access RAW_LATEX_DIR and RAW_PDF_DIR

def load_and_parse_documents():
    """
    Loads documents from specified directories, determines their type,
    and calls the appropriate parser.
    """
    all_parsed_texts = []

    # Process LaTeX files
    print(f"\n--- Processing LaTeX files from: {config.RAW_LATEX_DIR} ---")
    if not os.path.exists(config.RAW_LATEX_DIR):
        print(f"LaTeX directory not found: {config.RAW_LATEX_DIR}")
    else:
        for filename in os.listdir(config.RAW_LATEX_DIR):
            if filename.endswith(".tex"):
                file_path = os.path.join(config.RAW_LATEX_DIR, filename)
                print(f"Found LaTeX file: {file_path}")
                content = parse_latex_file(file_path)
                if content:
                    all_parsed_texts.append({"source": file_path, "type": "latex", "content": content})
            else:
                print(f"Skipping non-LaTeX file in LaTeX directory: {filename}")


    # Process PDF files
    print(f"\n--- Processing PDF files from: {config.RAW_PDF_DIR} ---")
    if not os.path.exists(config.RAW_PDF_DIR):
        print(f"PDF directory not found: {config.RAW_PDF_DIR}")
    else:
        for filename in os.listdir(config.RAW_PDF_DIR):
            if filename.endswith(".pdf"):
                file_path = os.path.join(config.RAW_PDF_DIR, filename)
                print(f"Found PDF file: {file_path}")
                # For now, assume general PDF. We can add logic to determine if it's math-heavy.
                # is_math_heavy = check_if_math_heavy(filename) # Placeholder for future logic
                is_math_heavy = "math" in filename.lower() # Simple heuristic for testing
                
                content = parse_pdf_file(file_path, is_math_heavy=is_math_heavy)
                if content:
                     all_parsed_texts.append({"source": file_path, "type": "pdf", "content": content})
            else:
                print(f"Skipping non-PDF file in PDF directory: {filename}")
    
    return all_parsed_texts

if __name__ == '__main__':
    # Create dummy files if they don't exist for testing
    dummy_tex_file = os.path.join(config.RAW_LATEX_DIR, "doctest_sample.tex")
    if not os.path.exists(config.RAW_LATEX_DIR): os.makedirs(config.RAW_LATEX_DIR)
    if not os.path.exists(dummy_tex_file):
        with open(dummy_tex_file, "w", encoding="utf-8") as f: f.write("Test LaTeX content for document_loader.")
        print(f"Created dummy file: {dummy_tex_file}")

    dummy_pdf_file = os.path.join(config.RAW_PDF_DIR, "doctest_sample.pdf")
    dummy_math_pdf_file = os.path.join(config.RAW_PDF_DIR, "doctest_sample_math.pdf")

    if not os.path.exists(config.RAW_PDF_DIR): os.makedirs(config.RAW_PDF_DIR)
    # For these PDF tests to run, you'd need to create small PDF files with these names.
    # For simplicity, we'll just check if they exist. If not, the PDF parsing part will be skipped by the loader.
    # You can create dummy PDFs (e.g. by saving an empty text file as PDF)
    if not os.path.exists(dummy_pdf_file):
        print(f"Warning: Dummy PDF {dummy_pdf_file} not found. PDF parsing test might be limited.")
    if not os.path.exists(dummy_math_pdf_file):
         print(f"Warning: Dummy PDF {dummy_math_pdf_file} not found. PDF parsing test might be limited.")


    print("Running document loader...")
    parsed_documents = load_and_parse_documents()
    print(f"\n--- Loaded and Parsed {len(parsed_documents)} documents ---")
    for doc in parsed_documents:
        print(f"Source: {doc['source']}, Type: {doc['type']}, Content Start: {doc['content'][:100]}...")
  