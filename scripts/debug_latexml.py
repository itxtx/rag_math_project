#!/usr/bin/env python3
"""
Debug script to examine LaTeXML output and diagnose text extraction issues
"""
import sys
import os
import xml.etree.ElementTree as ET
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion import latex_processor

def debug_latexml_output(latex_file_path):
    """Debug LaTeXML output for a given LaTeX file"""
    print(f"Debugging LaTeXML output for: {latex_file_path}")
    print("=" * 80)
    
    # Read the LaTeX content
    with open(latex_file_path, 'r', encoding='utf-8') as f:
        latex_content = f.read()
    
    # Find all proof environments in the original LaTeX
    proof_pattern = r'\\begin\{proof\}(.*?)\\end\{proof\}'
    proofs_in_latex = re.findall(proof_pattern, latex_content, re.DOTALL)
    print(f"\nFound {len(proofs_in_latex)} proof environments in LaTeX")
    
    # Process with LaTeXML
    xml_output = latex_processor.run_latexml_on_content(latex_content)
    
    if not xml_output:
        print("ERROR: LaTeXML returned no output")
        return
    
    # Save XML for inspection
    xml_file = latex_file_path.replace('.tex', '_latexml.xml')
    with open(xml_file, 'w', encoding='utf-8') as f:
        f.write(xml_output)
    print(f"\nSaved XML output to: {xml_file}")
    
    # Parse XML
    xml_output_clean = re.sub(r' xmlns="[^"]+"', '', xml_output, count=1)
    
    try:
        root = ET.fromstring(xml_output_clean)
    except ET.ParseError as e:
        print(f"ERROR: Could not parse XML: {e}")
        return
    
    # Find all proof elements
    proof_elements = root.findall(".//proof")
    theorem_elements = root.findall(".//theorem")
    
    print(f"\nFound {len(proof_elements)} proof elements in XML")
    print(f"Found {len(theorem_elements)} theorem elements in XML")
    
    # Examine each proof element
    for i, proof in enumerate(proof_elements):
        print(f"\n--- Proof {i+1} ---")
        print(f"Tag: {proof.tag}")
        print(f"Attributes: {proof.attrib}")
        
        # Show the raw XML for this proof
        proof_xml = ET.tostring(proof, encoding='unicode')
        print(f"\nRaw XML (first 500 chars):")
        print(proof_xml[:500])
        
        # Show title element if present
        title = proof.find('title')
        if title is not None:
            print(f"\nTitle element found:")
            print(f"  Text: '{title.text}'")
            print(f"  Children: {[child.tag for child in title]}")
            title_xml = ET.tostring(title, encoding='unicode')
            print(f"  Raw: {title_xml}")
        
        # Show all text content
        print(f"\nAll text nodes in proof:")
        for elem in proof.iter():
            if elem.text and elem.text.strip():
                print(f"  {elem.tag}: '{elem.text.strip()}'")
            if elem.tail and elem.tail.strip():
                print(f"  (tail after {elem.tag}): '{elem.tail.strip()}'")
        
        # Try different text extraction methods
        print(f"\nText extraction methods:")
        
        # Method 1: itertext()
        method1_text = ' '.join(proof.itertext()).strip()
        method1_text = re.sub(r'\s+', ' ', method1_text)
        print(f"\n1. itertext(): {method1_text[:200]}...")
        
        # Method 2: Custom extraction
        def extract_clean(elem, exclude_title=True):
            parts = []
            
            def extract_recursive(e, skip_title):
                if skip_title and e.tag == 'title':
                    return
                if e.text:
                    parts.append(e.text.strip())
                for child in e:
                    if not (exclude_title and child.tag == 'title'):
                        extract_recursive(child, False)
                    if child.tail:
                        parts.append(child.tail.strip())
            
            extract_recursive(elem, exclude_title)
            return ' '.join(p for p in parts if p)
        
        method2_text = extract_clean(proof)
        method2_text = re.sub(r'\s+', ' ', method2_text)
        print(f"\n2. Custom (no title): {method2_text[:200]}...")

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_latexml.py <latex_file>")
        sys.exit(1)
    
    latex_file = sys.argv[1]
    if not os.path.exists(latex_file):
        print(f"Error: File not found: {latex_file}")
        sys.exit(1)
    
    debug_latexml_output(latex_file)

if __name__ == "__main__":
    main()