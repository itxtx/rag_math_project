# src/data_ingestion/latex_processor.py
import subprocess
import os
import re
import xml.etree.ElementTree as ET

# Path to the master preamble relative to the project root
PREAMBLE_PATH = os.path.join('data', 'master_preamble.tex')

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
        with open("latexml_error.xml", "w", encoding="utf-8") as f:
            f.write(xml_string)
        print("Problematic XML saved to latexml_error.xml")
        return ""

def process_latex_document(input_text: str) -> str:
    """
    Processes a LaTeX document fragment using LaTeXML to expand all commands.
    This replaces the previous regex-based multi-pass system.
    """
    if not os.path.exists(PREAMBLE_PATH):
        raise FileNotFoundError(f"Master preamble not found at: {PREAMBLE_PATH}. Please create it.")

    # We process the entire input text as a fragment, assuming it's the body
    content_to_process = input_text

    print("INFO: Processing content with LaTeXML...")
    command = [
        "latexml",
        "--preload", PREAMBLE_PATH,  # Load all custom command definitions
        "--preamble", PREAMBLE_PATH, # Use the preamble file
        "--includestyles",          # Allow loading of .sty files
        "--xml",                    # Request XML output
        "-"                         # Read from stdin
    ]

    try:
        result = subprocess.run(
            command,
            input=content_to_process,
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True # Raise an error if LaTeXML fails
        )
        
        # The main logic is now parsing the XML output
        clean_text = _parse_xml_to_text(result.stdout)
        return clean_text

    except FileNotFoundError:
        print("ERROR: `latexml` command not found. Is LaTeXML installed and in your system's PATH?")
        return ""
    except subprocess.CalledProcessError as e:
        print(f"ERROR: LaTeXML failed to process the document fragment.")
        print(f"LaTeXML Stderr:\n{e.stderr}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred during LaTeXML processing: {e}")
        return ""

