# src/data_ingestion/latex_processor.py
import subprocess
import os
import re
import xml.etree.ElementTree as ET
import tempfile
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

def _parse_xml_to_text(xml_element) -> str:
    """A helper function to extract clean text from an XML element."""
    if xml_element is None:
        return ""
    # We use 'itertext()' to get all text from the element and its children.
    text_chunks = [text.strip() for text in xml_element.itertext() if text.strip()]
    # Join with newlines to preserve some paragraph structure
    return "\n\n".join(text_chunks)

def _create_full_latex_doc(content: str) -> str:
    """Wraps LaTeX content in a full document structure for processing."""
    return f"""
    \\documentclass{{article}}
    \\usepackage{{amsmath,amssymb,amsthm,bm}}
    {content}
    """

def run_latexml_on_content(latex_content: str) -> str:
    """
    Runs LaTeXML on a string of LaTeX content and returns the raw XML output.
    This is the core function that interacts with the LaTeXML command.
    """
    if not os.path.exists(PREAMBLE_PATH):
        raise FileNotFoundError(f"Master preamble not found at: {PREAMBLE_PATH}")

    _ensure_log_dir()
    temp_preamble_path = None
    temp_file_path = None
    latexml_log_path = None

    try:
        # Prepare the preamble with \renewcommand
        with open(PREAMBLE_PATH, 'r', encoding='utf-8') as f:
            preamble_content = re.sub(r'\\newcommand', r'\\renewcommand', f.read())
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False, encoding='utf-8') as temp_preamble_file:
            temp_preamble_file.write(preamble_content)
            temp_preamble_path = temp_preamble_file.name

        # LaTeXML works best with a full document structure.
        full_doc_content = _create_full_latex_doc(latex_content)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(full_doc_content)
            temp_file.flush()
            temp_file_path = temp_file.name
        
        print("INFO: Processing document with LaTeXML...")
        base_name = os.path.splitext(os.path.basename(temp_file_path))[0]
        latexml_log_path = os.path.join(LOG_DIR, f"{base_name}-latexml.log")

        command = [
            "latexml",
            "--log", latexml_log_path,
            "--preload", temp_preamble_path,
            "--includestyles",
            "--xml",
            "--nocomments",
            "--inputencoding=utf8",
            temp_file_path
        ]

        result = subprocess.run(
            command, capture_output=True, text=True, encoding='utf-8', check=True
        )
        return result.stdout

    except subprocess.CalledProcessError as e:
        print(f"ERROR: LaTeXML failed. A detailed log has been saved to: {latexml_log_path}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred during LaTeXML processing: {e}")
        return ""
    finally:
        # Robustly clean up the temporary files
        for path in [temp_file_path, temp_preamble_path]:
            if path and os.path.exists(path):
                os.unlink(path)