# src/data_ingestion/document_loader.py
import os
from typing import List, Dict, Optional, Literal, Set
from . import latex_parser 
from . import pdf_parser 
from src import config 

def load_processed_doc_filenames(log_file_path: str) -> Set[str]:
    """
    Loads the set of already processed document filenames from the log file.
    Each line in the log file is expected to be a filename.
    """
    processed_docs = set()
    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    processed_docs.add(line.strip())
            print(f"Loaded {len(processed_docs)} filenames from processed documents log: {log_file_path}")
        except Exception as e:
            print(f"Warning: Could not read processed documents log '{log_file_path}': {e}")
    else:
        print(f"Processed documents log not found at '{log_file_path}'. Will create if new documents are processed.")
    return processed_docs

def update_processed_docs_log(log_file_path: str, newly_processed_filenames: List[str]):
    """
    Appends newly processed document filenames to the log file.
    """
    if not newly_processed_filenames:
        return
    try:
        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir): # Check if log_dir is not empty string
            os.makedirs(log_dir, exist_ok=True)
            print(f"Created directory for processed documents log: {log_dir}")

        with open(log_file_path, 'a', encoding='utf-8') as f:
            for filename in newly_processed_filenames:
                f.write(f"{filename}\n")
        print(f"Appended {len(newly_processed_filenames)} filenames to processed documents log: {log_file_path}")
    except Exception as e:
        print(f"Warning: Could not update processed documents log '{log_file_path}': {e}")


def process_latex_documents(
        latex_dir: Optional[str] = None,
        processed_filenames_log: Optional[Set[str]] = None
    ) -> List[Dict]:
    """
    Processes all .tex files in the specified directory, skipping already processed ones.
    """
    if latex_dir is None:
        latex_dir = config.DATA_DIR_RAW_LATEX
    if processed_filenames_log is None: # Should ideally be passed in
        processed_filenames_log = set()

    parsed_latex_docs = []
    print(f"\n--- Processing LaTeX files from: {latex_dir} ---")
    if not os.path.isdir(latex_dir):
        print(f"LaTeX directory not found: {latex_dir}")
        return parsed_latex_docs

    # Create a single parser instance for all files
    parser = latex_parser.LatexToGraphParser()

    for filename in os.listdir(latex_dir):
        if filename.startswith('.'): # Skip hidden files like .DS_Store
            print(f"Skipping hidden file in LaTeX directory: {filename}")
            continue
        
        if filename in processed_filenames_log:
            print(f"Skipping already processed LaTeX file: {filename}")
            continue

        if filename.endswith(".tex"):
            file_path = os.path.join(latex_dir, filename)
            print(f"Processing LaTeX file: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    latex_content = f.read()
                
                # Extract structured nodes from the LaTeX content
                doc_id = os.path.splitext(filename)[0]
                parser.extract_structured_nodes(latex_content, doc_id)
                
                # Get the processed content from the graph
                nodes = parser.graph.nodes(data=True)
                if nodes:
                    # Combine all node texts into a single document
                    content = "\n\n".join(data['text'] for _, data in nodes)
                    parsed_latex_docs.append({
                        "doc_id": doc_id,
                        "source": file_path,
                        "filename": filename,
                        "original_type": "latex",
                        "parsed_content": content
                    })
                    
                    # Save parsed content
                    output_dir = config.DATA_DIR_PARSED_LATEX
                    if not os.path.exists(output_dir): os.makedirs(output_dir)
                    output_path = os.path.join(output_dir, f"{doc_id}_parsed.txt")
                    with open(output_path, "w", encoding="utf-8") as f_out:
                        f_out.write(content)
                    print(f"Saved parsed LaTeX content to: {output_path}")
                else:
                    print(f"Skipping LaTeX file due to no structured content found: {file_path}")
            except Exception as e:
                print(f"Error processing LaTeX file {file_path}: {e}")
        else:
            print(f"Skipping non-LaTeX file in LaTeX directory: {filename}")
            
    return parsed_latex_docs

def process_pdf_documents(
        pdf_dir: Optional[str] = None,
        pdf_processing_tool: Literal["general", "mathpix"] = "general",
        mathpix_app_id: Optional[str] = None,
        mathpix_app_key: Optional[str] = None,
        processed_filenames_log: Optional[Set[str]] = None
    ) -> List[Dict]:
    """
    Processes all .pdf files in the specified directory, skipping already processed ones.
    """
    if pdf_dir is None:
        pdf_dir = config.DATA_DIR_RAW_PDFS
    if processed_filenames_log is None:
        processed_filenames_log = set()

    parsed_pdf_docs = []
    print(f"\n--- Processing PDF files from: {pdf_dir} ---")
    if not os.path.isdir(pdf_dir):
        print(f"PDF directory not found: {pdf_dir}")
        return parsed_pdf_docs

    for filename in os.listdir(pdf_dir):
        if filename.startswith('.'): # Skip hidden files
            print(f"Skipping hidden file in PDF directory: {filename}")
            continue

        if filename in processed_filenames_log:
            print(f"Skipping already processed PDF file: {filename}")
            continue

        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            print(f"Processing PDF file: {file_path}")
            try:
                content = pdf_parser.parse_pdf_file(
                    file_path,
                    pdf_tool=pdf_processing_tool,
                    app_id=mathpix_app_id,
                    app_key=mathpix_app_key
                )
                if content and content.strip():
                    doc_id = os.path.splitext(filename)[0]
                    parsed_pdf_docs.append({
                        "doc_id": doc_id,
                        "source": file_path,
                        "filename": filename,
                        "original_type": "pdf",
                        "parsed_content": content
                    })
                    output_dir = config.DATA_DIR_PARSED_PDF
                    if not os.path.exists(output_dir): os.makedirs(output_dir)
                    output_path = os.path.join(output_dir, f"{doc_id}_parsed.txt")
                    with open(output_path, "w", encoding="utf-8") as f_out: f_out.write(content)
                    print(f"Saved parsed PDF content to: {output_path}")
                else:
                    print(f"Skipping PDF file due to empty parsed content: {file_path}")
            except Exception as e:
                print(f"Error processing PDF file {file_path}: {e}")
        else:
            print(f"Skipping non-PDF file in PDF directory: {filename}")
    return parsed_pdf_docs

def load_and_parse_documents(
    latex_dir: Optional[str] = None,
    pdf_dir: Optional[str] = None,
    pdf_math_tool: Literal["general", "mathpix"] = "general",
    process_latex: bool = True,
    process_pdfs: bool = True,
    processed_docs_log_path: Optional[str] = None
) -> List[Dict]:
    """
    Loads and parses documents from specified directories, skipping already processed ones.
    """
    if processed_docs_log_path is None:
        processed_docs_log_path = config.PROCESSED_DOCS_LOG_FILE
    
    already_processed_filenames = load_processed_doc_filenames(processed_docs_log_path)
    all_parsed_data = []

    if process_latex:
        parsed_latex_data = process_latex_documents(latex_dir, already_processed_filenames)
        all_parsed_data.extend(parsed_latex_data)

    if process_pdfs:
        app_id = getattr(config, 'MATHPIX_APP_ID', None)
        app_key = getattr(config, 'MATHPIX_APP_KEY', None)
        parsed_pdf_data = process_pdf_documents(
            pdf_dir,
            pdf_processing_tool=pdf_math_tool,
            mathpix_app_id=app_id,
            mathpix_app_key=app_key,
            processed_filenames_log=already_processed_filenames
        )
        all_parsed_data.extend(parsed_pdf_data)
    
    if not all_parsed_data:
        print("No new documents were found or parsed from any specified directory.")
    else:
        print(f"Total new documents parsed from all sources: {len(all_parsed_data)}")
        
    return all_parsed_data

if __name__ == '__main__':
    print("Running document_loader.py directly for testing persistence...")
    # Ensure config.PROCESSED_DOCS_LOG_FILE is set
    # Create dummy log file for testing
    dummy_log_file = os.path.join(config.DATA_DIR, "test_processed_log.txt")
    if os.path.exists(dummy_log_file): os.remove(dummy_log_file) # Clean start

    # Create dummy tex file
    os.makedirs(config.DATA_DIR_RAW_LATEX, exist_ok=True)
    dummy_tex_path = os.path.join(config.DATA_DIR_RAW_LATEX, "dummy_persist.tex")
    with open(dummy_tex_path, "w") as f: f.write("\\documentclass{article}\\begin{document}Test persist\\end{document}")

    print("\n--- First run (dummy_persist.tex should be processed) ---")
    parsed_docs1 = load_and_parse_documents(process_pdfs=False, processed_docs_log_path=dummy_log_file)
    newly_processed_filenames1 = [doc['filename'] for doc in parsed_docs1]
    update_processed_docs_log(dummy_log_file, newly_processed_filenames1)

    print("\n--- Second run (dummy_persist.tex should be skipped) ---")
    parsed_docs2 = load_and_parse_documents(process_pdfs=False, processed_docs_log_path=dummy_log_file)
    
    if os.path.exists(dummy_tex_path): os.remove(dummy_tex_path)
    if os.path.exists(dummy_log_file): os.remove(dummy_log_file)
    print("\nPersistence test finished.")
