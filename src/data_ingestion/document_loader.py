# src/data_ingestion/document_loader.py
import os
from typing import List, Dict, Optional, Literal
from . import latex_parser # Assuming latex_parser.py is in the same directory
from . import pdf_parser # Assuming pdf_parser.py is in the same directory
from src import config # For directory paths

def process_latex_documents(latex_dir: Optional[str] = None) -> List[Dict]:
    """
    Processes all .tex files in the specified directory.
    """
    if latex_dir is None:
        latex_dir = getattr(config, 'DATA_DIR_RAW_LATEX', 'data/raw_latex') # Fallback path

    parsed_latex_docs = []
    print(f"\n--- Processing LaTeX files from: {latex_dir} ---")
    if not os.path.isdir(latex_dir):
        print(f"LaTeX directory not found: {latex_dir}")
        return parsed_latex_docs

    for filename in os.listdir(latex_dir):
        if filename.endswith(".tex"):
            file_path = os.path.join(latex_dir, filename)
            print(f"Processing LaTeX file: {file_path}")
            try:
                # Assuming latex_parser.parse_latex_file returns the parsed text content
                content = latex_parser.parse_latex_file(file_path)
                if content and content.strip(): # Ensure content is not empty
                    doc_id = os.path.splitext(filename)[0]
                    parsed_latex_docs.append({
                        "doc_id": doc_id,
                        "source": file_path,
                        "original_type": "latex", # <<< CORRECTED TYPE
                        "parsed_content": content
                    })
                    # Optionally save parsed content to a file for review
                    output_dir = getattr(config, 'DATA_DIR_PARSED_LATEX', os.path.join(getattr(config, 'PARSED_CONTENT_DIR', 'data/parsed_content'), 'from_latex'))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    output_path = os.path.join(output_dir, f"{doc_id}_parsed.txt")
                    with open(output_path, "w", encoding="utf-8") as f_out:
                        f_out.write(content)
                    print(f"Saved parsed LaTeX content to: {output_path}")
                else:
                    print(f"Skipping LaTeX file due to empty parsed content: {file_path}")
            except Exception as e:
                print(f"Error processing LaTeX file {file_path}: {e}")
        elif not filename.startswith('.'): # Ignore hidden files like .DS_Store
            print(f"Skipping non-LaTeX file in LaTeX directory: {filename}")
            
    return parsed_latex_docs

def process_pdf_documents(pdf_dir: Optional[str] = None,
                          pdf_processing_tool: Literal["general", "mathpix"] = "general",
                          mathpix_app_id: Optional[str] = None,
                          mathpix_app_key: Optional[str] = None) -> List[Dict]:
    """
    Processes all .pdf files in the specified directory.
    """
    if pdf_dir is None:
        pdf_dir = getattr(config, 'DATA_DIR_RAW_PDFS', 'data/raw_pdfs') # Fallback path

    parsed_pdf_docs = []
    print(f"\n--- Processing PDF files from: {pdf_dir} ---")
    if not os.path.isdir(pdf_dir):
        print(f"PDF directory not found: {pdf_dir}")
        return parsed_pdf_docs

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            print(f"Processing PDF file: {file_path}")
            try:
                # Assuming pdf_parser.parse_pdf_file handles different tools
                content = pdf_parser.parse_pdf_file(
                    file_path,
                    pdf_tool=pdf_processing_tool,
                    app_id=mathpix_app_id,
                    app_key=mathpix_app_key
                )
                if content and content.strip(): # Ensure content is not empty
                    doc_id = os.path.splitext(filename)[0]
                    parsed_pdf_docs.append({
                        "doc_id": doc_id,
                        "source": file_path,
                        "original_type": "pdf", # Correct type for PDFs
                        "parsed_content": content
                    })
                    # Optionally save parsed content
                    output_dir = getattr(config, 'DATA_DIR_PARSED_PDF', os.path.join(getattr(config, 'PARSED_CONTENT_DIR', 'data/parsed_content'), 'from_pdf'))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    output_path = os.path.join(output_dir, f"{doc_id}_parsed.txt")
                    with open(output_path, "w", encoding="utf-8") as f_out:
                        f_out.write(content)
                    print(f"Saved parsed PDF content to: {output_path}")
                else:
                    print(f"Skipping PDF file due to empty parsed content: {file_path}")
            except Exception as e:
                print(f"Error processing PDF file {file_path}: {e}")
        elif not filename.startswith('.'): # Ignore hidden files
            print(f"Skipping non-PDF file in PDF directory: {filename}")
    return parsed_pdf_docs

def load_and_parse_documents(
    latex_dir: Optional[str] = None,
    pdf_dir: Optional[str] = None,
    pdf_math_tool: Literal["general", "mathpix"] = "general",
    process_latex: bool = True, # Flag to control LaTeX processing
    process_pdfs: bool = True   # Flag to control PDF processing
) -> List[Dict]:
    """
    Loads and parses documents from specified directories.
    By default, processes both LaTeX and PDF files.
    """
    all_parsed_data = []

    if process_latex:
        # Get Mathpix keys from config if not passed directly (though not used by LaTeX parser)
        # This is more relevant for the PDF parser if it were to use them.
        # For consistency, we can read them here.
        # app_id = getattr(config, 'MATHPIX_APP_ID', None)
        # app_key = getattr(config, 'MATHPIX_APP_KEY', None)
        
        parsed_latex_data = process_latex_documents(latex_dir)
        all_parsed_data.extend(parsed_latex_data)

    if process_pdfs:
        app_id = getattr(config, 'MATHPIX_APP_ID', None)
        app_key = getattr(config, 'MATHPIX_APP_KEY', None)
        parsed_pdf_data = process_pdf_documents(
            pdf_dir,
            pdf_processing_tool=pdf_math_tool,
            mathpix_app_id=app_id,
            mathpix_app_key=app_key
        )
        all_parsed_data.extend(parsed_pdf_data)
    
    if not all_parsed_data:
        print("No documents were found or parsed from any specified directory.")
    else:
        print(f"Total documents parsed from all sources: {len(all_parsed_data)}")
        
    return all_parsed_data

if __name__ == '__main__':
    # Example of direct execution for testing
    print("Running document_loader.py directly for testing...")

    # Test LaTeX processing
    # Ensure config.DATA_DIR_RAW_LATEX is set or provide a path
    # Create a dummy tex file if needed in your data/raw_latex directory
    # e.g., data/raw_latex/sample.tex
    # \documentclass{article} \begin{document} Hello LaTeX. \section{Intro} Some text. \end{document}
    
    # To run this, you might need to adjust config paths or create dummy files.
    # For example, set config.DATA_DIR_RAW_LATEX and config.DATA_DIR_RAW_PDFS
    # or ensure the default paths 'data/raw_latex' and 'data/raw_pdfs' exist
    # relative to where this script is run from (or use absolute paths in config).

    # Assuming config paths are set or default paths exist:
    # parsed_docs = load_and_parse_documents(process_pdfs=False) # Test only LaTeX
    parsed_docs = load_and_parse_documents() # Test both
    
    if parsed_docs:
        print(f"\n--- Summary of parsed documents ({len(parsed_docs)}) ---")
        for i, doc_data in enumerate(parsed_docs):
            print(f"Doc {i+1}:")
            print(f"  ID: {doc_data.get('doc_id')}")
            print(f"  Source: {doc_data.get('source')}")
            print(f"  Type: {doc_data.get('original_type')}")
            print(f"  Content (first 100 chars): {doc_data.get('parsed_content', '')[:100]}...")
    else:
        print("No documents were parsed in the test run.")

