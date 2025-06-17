# src/data_ingestion/document_loader.py
import os
from typing import List, Dict, Set

from src import config

def load_processed_doc_filenames(log_file_path: str) -> Set[str]:
    """Loads the set of already processed document filenames from the log file."""
    processed_docs = set()
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r', encoding='utf-8') as f:
            processed_docs.update(line.strip() for line in f)
        print(f"Loaded {len(processed_docs)} filenames from processed documents log: {log_file_path}")
    return processed_docs

def update_processed_docs_log(log_file_path: str, newly_processed_filenames: List[str]):
    """Appends newly processed document filenames to the log file."""
    if not newly_processed_filenames:
        return
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    with open(log_file_path, 'a', encoding='utf-8') as f:
        for filename in newly_processed_filenames:
            f.write(f"{filename}\n")
    print(f"Appended {len(newly_processed_filenames)} filenames to processed documents log.")

def load_new_documents() -> List[Dict]:
    """
    Finds new LaTeX and PDF documents and loads their raw content.
    This function NO LONGER PARSES. It only loads.
    """
    already_processed = load_processed_doc_filenames(config.PROCESSED_DOCS_LOG_FILE)
    new_docs_to_process = []

    # --- Process LaTeX Files ---
    print(f"\n--- Checking for new LaTeX files in: {config.DATA_DIR_RAW_LATEX} ---")
    if os.path.isdir(config.DATA_DIR_RAW_LATEX):
        for filename in os.listdir(config.DATA_DIR_RAW_LATEX):
            if filename.startswith('.') or not filename.endswith(".tex"):
                continue
            if filename in already_processed:
                continue

            file_path = os.path.join(config.DATA_DIR_RAW_LATEX, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                new_docs_to_process.append({
                    "doc_id": os.path.splitext(filename)[0],
                    "source": file_path,
                    "filename": filename,
                    "type": "latex",
                    "raw_content": raw_content
                })
                print(f"Found new LaTeX document: {filename}")
            except Exception as e:
                print(f"Error reading LaTeX file {file_path}: {e}")

    # --- Add PDF processing logic here if needed, similar to above ---
    # For now, focusing on the LaTeX issue.

    if not new_docs_to_process:
        print("No new documents found.")
    else:
        print(f"Found {len(new_docs_to_process)} total new documents.")

    return new_docs_to_process