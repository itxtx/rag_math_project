# src/data_ingestion/latex_parser.py
import os
import re
from pylatexenc.latexwalker import LatexWalker, LatexCharsNode, LatexCommentNode, LatexGroupNode, LatexMacroNode, LatexMathNode, LatexEnvironmentNode, get_default_latex_context_db
from pylatexenc.latex2text import LatexNodes2Text
from pylatexenc.macrospec import MacroSpec, MacroStandardArgsParser, LatexContextDb

from src import config

def custom_latex_to_text(latex_str: str) -> str:
    """
    Converts LaTeX string to a cleaner text representation.
    Prioritizes content within the 'document' environment.
    """
    print(f"DEBUG custom_latex_to_text: Input LaTeX string (first 200 chars): '{latex_str[:200]}...'")
    if not latex_str.strip():
        print("DEBUG custom_latex_to_text: LaTeX string is empty or whitespace.")
        return ""
    
    lw = LatexWalker(latex_str) 
    parsing_result = lw.get_latex_nodes()

    if parsing_result is None:
        print("DEBUG custom_latex_to_text: LatexWalker.get_latex_nodes() returned None directly.")
        nodelist = []
    else:
        nodelist, pos_end, len_end = parsing_result
        if nodelist is None:
            print("DEBUG custom_latex_to_text: Nodelist from get_latex_nodes() is None.")
            nodelist = []
    
    print(f"DEBUG custom_latex_to_text: Initial nodelist length: {len(nodelist)}")
    if len(nodelist) > 0:
        print(f"DEBUG custom_latex_to_text: First node type: {type(nodelist[0]).__name__ if nodelist else 'N/A'}")


    class CustomLatexNodes2Text(LatexNodes2Text):
        def __init__(self, **kwargs):
            super().__init__(math_mode="verbatim", **kwargs) 
            
            if not hasattr(self, 'latex_context_db') or self.latex_context_db is None:
                print("WARN: self.latex_context_db not set by superclass or is None. Initializing a new default one.")
                self.latex_context_db = get_default_latex_context_db()
            
            print(f"DEBUG __init__: CustomLatexNodes2Text initialized. Using default handlers for sectioning commands.")
            # Preamble macros are handled in convert_node by returning ""


        def convert_node(self, node):
            if node is None:
                return ""
            
            if node.isNodeType(LatexCharsNode):
                return node.chars

            if node.isNodeType(LatexMathNode):
                return node.latex_verbatim()

            if node.isNodeType(LatexMacroNode):
                print(f"DEBUG convert_node: Encountered LatexMacroNode with macroname: '{node.macroname}'")
                
                if node.macroname in ['textit', 'textbf', 'emph']:
                    if node.nodeargs and len(node.nodeargs) > 0 and \
                       node.nodeargs[0] is not None and hasattr(node.nodeargs[0], 'nodelist'):
                        return self.nodelist_to_text(node.nodeargs[0].nodelist)
                    return "" 

                macros_to_remove_entirely = [
                    'title', 'author', 'date', 'maketitle', 
                    'documentclass', 'usepackage',
                    'label', 'ref', 'cite', 'pagestyle', 'thispagestyle',
                    'includegraphics', 'figureheight', 'figurewidth', 'caption',
                    'tableofcontents', 'listoffigures', 'listoftables',
                    'appendix', 'bibliographystyle', 'bibliography',
                    'hspace', 'vspace', 'hfill', 'vfill', 'centering', 'noindent', 'indent',
                    'newpage', 'clearpage', 'linebreak', 'nolinebreak',
                    'setlength', 'addtolength', 'setcounter', 'addtocounter',
                    'selectfont', 'item',
                    'newtheorem', 'newenvironment', 'renewenvironment',
                    'newcommand', 'renewcommand', 'providecommand',
                    'DeclareMathOperator', 'DeclareRobustCommand',
                    'geometry', 'setstretch', 'linespread',
                    'hyphenation', 'url', 'href', 'graphicspath', 'input', 'include',
                    'RequirePackage', 'LoadClass', 'ProcessOptions', 'PassOptionsToPackage',
                    'newif', 'ifdefined', 'ifx', 'fi', 'else', 'let', 'def', 'edef', 'gdef', 'xdef',
                    'AtBeginDocument', 'AtEndDocument', 'AtEndOfClass', 'AtEndOfPackage',
                    'ExplSyntaxOn', 'ExplSyntaxOff',
                    'ProvidesPackage', 'NeedsTeXFormat'
                ]
                if node.macroname in macros_to_remove_entirely:
                    print(f"DEBUG convert_node: Removing macro '{node.macroname}' and its output by returning empty string.")
                    return "" 
                
                print(f"DEBUG convert_node: For macro '{node.macroname}', calling super().convert_node() for dispatch.")
                return super().convert_node(node)


            if node.isNodeType(LatexCommentNode):
                return ""
            
            if node.isNodeType(LatexEnvironmentNode):
                # This is key: if we find the 'document' environment, we ONLY process its content.
                # This check in convert_node is a fallback if the top-level filtering in custom_latex_to_text fails.
                if node.environmentname == 'document':
                    print("DEBUG convert_node: Processing 'document' environment content (called from within).")
                    return self.nodelist_to_text(node.nodelist)
                
                environments_to_process_content = [
                    'abstract', 'center', 
                    'itemize', 'enumerate', 'description',
                    'figure', 'table', 
                    'theorem', 'lemma', 'proof', 'definition', 'corollary', 'example', 'remark'
                ]
                if node.environmentname in environments_to_process_content:
                    print(f"DEBUG convert_node: Processing content of environment '{node.environmentname}'.")
                    return "\n" + self.nodelist_to_text(node.nodelist) + "\n"

                environments_to_remove = ['comment']
                if node.environmentname in environments_to_remove:
                    print(f"DEBUG convert_node: Removing environment '{node.environmentname}'.")
                    return ""
                
                print(f"DEBUG convert_node: Environment '{node.environmentname}' not explicitly handled, calling super.")
                return super().convert_node(node)

            if node.isNodeType(LatexGroupNode):
                return self.nodelist_to_text(node.nodelist)

            return super().convert_node(node)

    # --- Logic to isolate 'document' environment content ---
    document_node = None
    if nodelist: # Ensure nodelist is not empty before iterating
        for i, node in enumerate(nodelist):
            #print(f"DEBUG custom_latex_to_text: Top-level node {i} type: {type(node).__name__}")
            if node.isNodeType(LatexEnvironmentNode) and node.environmentname == 'document':
                document_node = node
                #print(f"DEBUG custom_latex_to_text: Found 'document' environment at top-level node index {i}.")
                break
    
    converter = CustomLatexNodes2Text()
    text_content = "" # Default to empty

    if document_node:
        #print("DEBUG custom_latex_to_text: Processing ONLY content of 'document' environment.")
        text_content = converter.nodelist_to_text(document_node.nodelist)
    else:
        # If no 'document' environment is found at the top level, it's likely a fragment
        # or a malformed document for our purposes. Return empty to avoid processing preamble.
        print("WARNING custom_latex_to_text: No top-level 'document' environment found. Returning empty string to avoid processing preamble as content.")
        # Optionally, you could process the full nodelist here if you expect valid fragments,
        # but it's safer to be strict for full documents.
        # text_content = converter.nodelist_to_text(nodelist) # This would process preamble if no document env
    # --- End of logic ---
    
    if text_content: # Ensure text_content is a string before regex
        text_content = re.sub(r'(\n\s*){3,}', '\n\n', text_content)
        text_content = text_content.strip()
    
    #print(f"DEBUG custom_latex_to_text: Final text content (first 200 chars): '{text_content[:200]}...'")
    return text_content


def parse_latex_file(file_path: str) -> str:
    print(f"Attempting to parse LaTeX file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            latex_content = f.read()
        
        if not latex_content.strip():
            print(f"LaTeX file is empty or contains only whitespace: {file_path}")
            return ""
            
        text_representation = custom_latex_to_text(latex_content)
        
        # This warning is now less critical if custom_latex_to_text returns "" for no document env
        if not text_representation.strip() and latex_content.strip():
            print(f"INFO: custom_latex_to_text returned empty/whitespace for non-empty file (likely no 'document' env found): {file_path}")
            
        return text_representation

    except FileNotFoundError:
        print(f"ERROR: LaTeX file not found: {file_path}")
        return ""
    except Exception as e:
        print(f"DEBUG: Error parsing LaTeX file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return ""

if __name__ == '__main__':
    class DummyConfig:
        RAW_LATEX_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw_latex')
        PARSED_LATEX_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'parsed_content', 'from_latex')

    cfg = DummyConfig() 
    
    raw_latex_dir_to_use = cfg.RAW_LATEX_DIR
    if hasattr(config, 'DATA_DIR_RAW_LATEX') and os.path.exists(config.DATA_DIR_RAW_LATEX):
        raw_latex_dir_to_use = config.DATA_DIR_RAW_LATEX

    parsed_latex_output_dir_to_use = cfg.PARSED_LATEX_OUTPUT_DIR
    if hasattr(config, 'DATA_DIR_PARSED_LATEX') and os.path.exists(config.DATA_DIR_PARSED_LATEX):
         parsed_latex_output_dir_to_use = config.DATA_DIR_PARSED_LATEX
    elif hasattr(config, 'PARSED_CONTENT_DIR'):
         parsed_latex_output_dir_to_use = os.path.join(config.PARSED_CONTENT_DIR, 'from_latex')


    print(f"--- Processing all .tex files in: {raw_latex_dir_to_use} ---")
    
    if not os.path.exists(raw_latex_dir_to_use):
        print(f"Directory not found: {raw_latex_dir_to_use}. Please create it and add .tex files.")
        exit()
    
    if not os.path.exists(parsed_latex_output_dir_to_use):
        os.makedirs(parsed_latex_output_dir_to_use)
        print(f"Created output directory: {parsed_latex_output_dir_to_use}")

    tex_files_in_dir = [f for f in os.listdir(raw_latex_dir_to_use) if f.endswith(".tex")]
    if not tex_files_in_dir:
        dummy_tex_filename_main = "main_sample.tex"
        dummy_tex_file_main = os.path.join(raw_latex_dir_to_use, dummy_tex_filename_main)
        with open(dummy_tex_file_main, "w", encoding="utf-8") as f:
            f.write("\\documentclass{article}\n")
            f.write("\\title{Sample for Main Execution}\n")
            f.write("\\author{The Parser}\n")
            f.write("\\date{\\today}\n")
            f.write("\\begin{document}\n")
            f.write("\\maketitle\n")
            f.write("\\section{Test Section in Main}\n")
            f.write("Content for main execution test. Hello World from __main__! $E=mc^2 + \\alpha$.\n")
            f.write("\\subsection{A Subsection}\n")
            f.write("Some more text here with \\textbf{bold} and \\textit{italic} styles.\n")
            f.write("\\begin{itemize}\n")
            f.write("  \\item First item.\n")
            f.write("  \\item Second item: $x_1, x_2$.\n")
            f.write("\\end{itemize}\n")
            f.write("\\end{document}\n")
        print(f"Created dummy file for main execution: {dummy_tex_file_main}")
        tex_files_in_dir = [dummy_tex_filename_main]

    processed_count = 0
    for filename in tex_files_in_dir:
        file_path = os.path.join(raw_latex_dir_to_use, filename)
        print(f"\n--- Parsing: {filename} ---")
        parsed_content = parse_latex_file(file_path)
        
        if parsed_content:
            print(f"--- Parsed Content from {filename} (first 500 chars) ---")
            print(parsed_content[:500] + "..." if len(parsed_content) > 500 else parsed_content)
            
            output_filename = os.path.splitext(filename)[0] + "_parsed.txt"
            output_path = os.path.join(parsed_latex_output_dir_to_use, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as f_out:
                f_out.write(parsed_content)
            print(f"Saved parsed LaTeX content to: {output_path}")
            processed_count += 1
        else:
            print(f"Could not parse or empty content for: {filename}")
            
    if processed_count == 0 and tex_files_in_dir:
        print("\nWARNING: No .tex files were successfully parsed, though files were present.")
    elif not tex_files_in_dir:
         print(f"\nNo .tex files found in {raw_latex_dir_to_use} to process for __main__ demo.")

    print(f"\n--- Finished processing. Parsed {processed_count} .tex files. ---")

