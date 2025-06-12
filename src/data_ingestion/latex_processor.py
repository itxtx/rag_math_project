# src/data_ingestion/latex_processor.py
import subprocess
import os
import re
import xml.etree.ElementTree as ET
import tempfile
from datetime import datetime

PREAMBLE_PATH = os.path.join('data', 'master_preamble.tex')
LOG_DIR = os.path.join('data', 'latex_temp', 'logs')

def _ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)

def _extract_preamble_commands(full_preamble_content: str) -> str:
    """
    Extracts only the preamble commands (between \documentclass and \begin{document}).
    """
    # Use raw string to avoid escape issues
    match = re.search(r'\\documentclass\{.*?\}(.*?)\\begin\{document\}', full_preamble_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback if the preamble file is just a list of commands
    return full_preamble_content

def _extract_document_body(full_latex_content: str) -> str:
    """
    Extracts only the content between \\begin{document} and \\end{document}.
    """
    # Use raw string to avoid escape issues
    match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', full_latex_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return full_latex_content # Fallback for fragments

def run_latexml_on_content(latex_content: str) -> str:
    """
    Runs LaTeXML by intelligently assembling a clean, valid document from a master
    preamble and the body of the input content. This is the most robust method.
    """
    if not os.path.exists(PREAMBLE_PATH):
        raise FileNotFoundError(f"Master preamble not found at: {PREAMBLE_PATH}")

    _ensure_log_dir()
    temp_file_path = None
    latexml_log_path = None

    try:
        # 1. Read the master preamble file.
        with open(PREAMBLE_PATH, 'r', encoding='utf-8') as f:
            master_preamble_full = f.read()

        # 2. Extract ONLY the commands from the master preamble.
        preamble_commands = _extract_preamble_commands(master_preamble_full)
        # Replace \newcommand with \renewcommand to avoid "already defined" errors.
        # Use raw string for the replacement pattern
        preamble_commands = re.sub(r'\\newcommand', r'\\renewcommand', preamble_commands)

        # 3. Extract ONLY the body content from the source document.
        document_body = _extract_document_body(latex_content)

        # 4. Assemble a single, complete, and valid LaTeX document string.
        full_doc_to_process = f"""
\\documentclass{{article}}
{preamble_commands}
\\begin{{document}}
{document_body}
\\end{{document}}
"""
        # 5. Write this new document to a temporary file.
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(full_doc_to_process)
            temp_file.flush()
            temp_file_path = temp_file.name

        print("INFO: Processing document with LaTeXML...")
        base_name = os.path.splitext(os.path.basename(temp_file_path))[0]
        latexml_log_path = os.path.join(LOG_DIR, f"{base_name}-latexml.log")

        # 6. Run LaTeXML on the single, well-formed temporary file.
        command = [
            "latexml",
            "--log", latexml_log_path,
            "--includestyles",
            "--xml",
            "--nocomments",
            "--inputencoding=utf8",
            temp_file_path
        ]

        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(
            command, capture_output=True, text=True, encoding='utf-8', check=True
        )
        
        # Debug output
        if result.stdout:
            print(f"LaTeXML stdout length: {len(result.stdout)} characters")
            if len(result.stdout) < 500:
                print(f"LaTeXML stdout: {result.stdout}")
        if result.stderr:
            print(f"LaTeXML stderr: {result.stderr[:500]}")
        
        return result.stdout

    except subprocess.CalledProcessError as e:
        print(f"ERROR: LaTeXML failed with exit code {e.returncode}")
        if e.stdout:
            print(f"LaTeXML stdout: {e.stdout[:500]}...")
        if e.stderr:
            print(f"LaTeXML stderr: {e.stderr[:500]}...")
        print(f"A detailed log has been saved to: {latexml_log_path}")
        print("Please check this log file for the specific error from LaTeXML.")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred during LaTeXML processing: {e}")
        import traceback
        traceback.print_exc()
        return ""
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)