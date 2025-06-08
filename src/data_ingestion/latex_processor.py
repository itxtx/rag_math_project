# src/data_ingestion/latex_processor.py
import subprocess
import os
import re
import xml.etree.ElementTree as ET
import tempfile
import shutil
from datetime import datetime

# Path to the master preamble relative to the project root
PREAMBLE_PATH = os.path.join('data', 'master_preamble.tex')
# Path to store LaTeX logs
LOG_DIR = os.path.join('data', 'latex_temp', 'logs')

def _ensure_log_dir():
    """Ensure the log directory exists."""
    os.makedirs(LOG_DIR, exist_ok=True)

def _get_log_path(filename: str) -> str:
    """Get the path for a log file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = os.path.splitext(os.path.basename(filename))[0]
    return os.path.join(LOG_DIR, f"{base_name}_{timestamp}.log")

def _parse_xml_to_text(xml_string: str) -> str:
    """A helper function to extract clean text from LaTeXML's output."""
    if not xml_string:
        return ""
    try:
        # LaTeXML often wraps output in a specific namespace. We need to handle it.
        # This removes the namespace for easier tag matching.
        xml_string = re.sub(r' xmlns="[^"]+"', '', xml_string, count=1)
        root = ET.fromstring(xml_string)
        
        # We use 'itertext()' to get all text from elements and their children.
        text_chunks = [text.strip() for text in root.itertext() if text.strip()]
        
        # Join with newlines to preserve some paragraph structure
        return "\n\n".join(text_chunks)
    except ET.ParseError as e:
        print(f"ERROR: Could not parse LaTeXML output. Error: {e}")
        # Log the problematic XML for debugging
        xml_log_path = os.path.join(LOG_DIR, f"latexml_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml")
        with open(xml_log_path, "w", encoding="utf-8") as f:
            f.write(xml_string)
        print(f"Problematic XML saved to {xml_log_path}")
        return ""

def _extract_document_content(input_text: str) -> str:
    """Extract the content between \begin{document} and \end{document}."""
    # Find the document content
    doc_match = re.search(r'\\begin{document}(.*?)\\end{document}', input_text, re.DOTALL)
    if not doc_match:
        return input_text
    return doc_match.group(1).strip()

def _prepare_preamble() -> str:
    """Prepare the preamble by converting \newcommand to \renewcommand."""
    if not os.path.exists(PREAMBLE_PATH):
        raise FileNotFoundError(f"Master preamble not found at: {PREAMBLE_PATH}")
    
    with open(PREAMBLE_PATH, 'r', encoding='utf-8') as f:
        preamble = f.read()
    
    # Convert \newcommand to \renewcommand
    preamble = re.sub(r'\\newcommand', r'\\renewcommand', preamble)
    
    # Create a temporary file for the modified preamble
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as temp_file:
        temp_file.write(preamble)
        temp_preamble_path = temp_file.name
    
    return temp_preamble_path

def process_latex_document(input_text: str) -> str:
    """
    Processes a LaTeX document using LaTeXML to expand all commands.
    This replaces the previous regex-based multi-pass system.
    """
    if not os.path.exists(PREAMBLE_PATH):
        raise FileNotFoundError(f"Master preamble not found at: {PREAMBLE_PATH}. Please create it.")

    # Ensure log directory exists and initialize paths for robust cleanup
    _ensure_log_dir()
    temp_preamble_path = None
    temp_file_path = None

    try:
        content = _extract_document_content(input_text)
        temp_preamble_path = _prepare_preamble()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False, encoding='utf-8') as temp_file:
            temp_file.write("\\documentclass{article}\n")
            temp_file.write("\\usepackage{amsmath,amssymb,amsthm,bm}\n")
            temp_file.write("\\begin{document}\n")
            temp_file.write(content)
            temp_file.write("\n\\end{document}\n")
            temp_file.flush() # Essential: Ensures file is written to disk
            temp_file_path = temp_file.name

        print("INFO: Processing content with LaTeXML...")

        # --- FIX FOR LOG FILE LOCATION ---
        # Create a specific path for the LaTeXML diagnostic log file
        base_name = os.path.splitext(os.path.basename(temp_file_path))[0]
        latexml_log_path = os.path.join(LOG_DIR, f"{base_name}-latexml.log")

        command = [
            "latexml",
            "--log", latexml_log_path,      # ❗ Direct LaTeXML's own log to the correct directory
            "--preload", temp_preamble_path,
            "--includestyles",
            "--xml",
            "--nocomments",
            "--inputencoding=utf8",
            temp_file_path
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True
        )
        
        
        # This is optional if the main latexml log is sufficient
        overview_log_path = os.path.join(LOG_DIR, f"{base_name}-overview.log")
        with open(overview_log_path, 'w', encoding='utf-8') as f:
            f.write(f"--- LaTeXML Command ---\n{' '.join(command)}\n\n")
            f.write("--- STDOUT ---\n")
            f.write(result.stdout)
            f.write("\n\n--- STDERR ---\n")
            f.write(result.stderr)
        
        return _parse_xml_to_text(result.stdout)

    except subprocess.CalledProcessError as e:
        # This block will now be less critical since the persistent log is saved
        # automatically by LaTeXML into your LOG_DIR.
        print(f"ERROR: LaTeXML failed. A detailed log has been saved to:")
        print(f"  {latexml_log_path}")
        return ""
    except FileNotFoundError:
        print("ERROR: `latexml` command not found. Is LaTeXML installed and in your system's PATH?")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred during LaTeXML processing: {e}")
        return ""
    finally:
        # Robustly clean up the temporary INPUT files
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if temp_preamble_path and os.path.exists(temp_preamble_path):
            os.unlink(temp_preamble_path)