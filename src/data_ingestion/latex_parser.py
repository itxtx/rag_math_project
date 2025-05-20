# src/data_ingestion/latex_parser.py
import os
import re
from pylatexenc.latexwalker import LatexWalker, LatexCharsNode, LatexCommentNode, LatexGroupNode, LatexMacroNode, LatexMathNode, LatexEnvironmentNode
from pylatexenc.latex2text import LatexNodes2Text
from pylatexenc.macrospec import MacroSpec # For advanced context db modification if needed

from src import config

def custom_latex_to_text(latex_str: str) -> str:
    """
    Converts LaTeX string to a cleaner text representation,
    attempting to preserve math and structure.
    """
    if not latex_str.strip(): 
        return ""

    lw = LatexWalker(latex_str)
    nodelist, _, _ = lw.get_latex_nodes()
    
    if nodelist is None: 
        nodelist = []

    class CustomLatexNodes2Text(LatexNodes2Text):
        def __init__(self, **kwargs):
            super().__init__(math_mode=LatexNodes2Text.MATH_MODE_VERBATIM, **kwargs)
            # For more fine-grained control, you can modify the latex_context_db
            # For example, to remove default handlers for sectioning if needed:
            # self.latex_context_db.remove_macro('section') # And others
            # Then ensure your latex_macro_section is used.
            # However, pylatexenc *should* prefer subclass methods.

        def _format_structural_macro(self, node, macroname):
            arg_text = ""
            if node.nodeargs and node.nodeargs[0] and hasattr(node.nodeargs[0], 'nodelist'): 
                arg_text = self.nodelist_to_text(node.nodeargs[0].nodelist)
            return f"\n\n\\{macroname}{{{arg_text}}}\n" 

        def latex_macro_section(self, node, **kwargs): # `node` is LatexMacroNode
            return self._format_structural_macro(node, 'section')

        def latex_macro_subsection(self, node, **kwargs):
            return self._format_structural_macro(node, 'subsection')

        def latex_macro_subsubsection(self, node, **kwargs):
            return self._format_structural_macro(node, 'subsubsection')

        def latex_macro_paragraph(self, node, **kwargs): 
            return self._format_structural_macro(node, 'paragraph')
        
        def convert_node(self, node):
            if node is None:
                return ""
            
            if node.isNodeType(LatexCharsNode): 
                return node.chars

            if node.isNodeType(LatexMathNode):
                return node.latex_verbatim() 

            if node.isNodeType(LatexMacroNode):
                # Check if we have a specific handler for this macro in our class
                # (e.g., latex_macro_section). pylatexenc's dispatch mechanism
                # should call these specific handlers if they exist on `self`.
                # So, if execution reaches this generic LatexMacroNode check in *our*
                # convert_node, it means it's not one of our specifically handled ones.

                # Handle macros we want to transform text from (like \textit)
                if node.macroname in ['textit', 'textbf', 'emph']: 
                    if node.nodeargs and node.nodeargs[0] and hasattr(node.nodeargs[0], 'nodelist'):
                        return self.nodelist_to_text(node.nodeargs[0].nodelist)
                    return "" 
                
                # Handle macros we want to remove entirely
                macros_to_remove = [
                    'documentclass', 'usepackage', 
                    'title', 'author', 'date', 'maketitle',
                    'label', 'ref', 'cite', 'pagestyle', 'thispagestyle',
                    'includegraphics', 'figureheight', 'figurewidth', 'caption',
                    'tableofcontents', 'listoffigures', 'listoftables',
                    'appendix', 'bibliographystyle', 'bibliography'
                ]
                if node.macroname in macros_to_remove:
                    return ""
                
                # For ALL other macros (those not section-like, not textit-like, not in remove list),
                # let the base class's convert_node attempt to handle it.
                # The base class might have its own specific handlers or a default.
                # Its default for unknown macros (latex_default_macro) returns '',
                # but it might know some standard macros.
                return super().convert_node(node)


            if node.isNodeType(LatexCommentNode):
                return "" 
            
            if node.isNodeType(LatexEnvironmentNode):
                if node.environmentname == 'document':
                     return self.nodelist_to_text(node.nodelist)
                # Math environments are handled by LatexMathNode due to MATH_MODE_VERBATIM
                # For other environments, we can choose to process their content or delegate
                if node.environmentname in ['abstract', 'center', 'figure', 'table']: # Examples
                     return self.nodelist_to_text(node.nodelist)
                # Fallback for other environments
                return super().convert_node(node)


            if node.isNodeType(LatexGroupNode): 
                return self.nodelist_to_text(node.nodelist)

            # Default fallback for any other node type not handled above
            return super().convert_node(node)

    converter = CustomLatexNodes2Text()
    text_content = converter.nodelist_to_text(nodelist)
    
    text_content = re.sub(r'(\n\s*){3,}', '\n\n', text_content) 
    text_content = text_content.strip()
    return text_content

def parse_latex_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            latex_content = f.read()
        if not latex_content.strip(): 
            return ""
        text_representation = custom_latex_to_text(latex_content)
        return text_representation
    except FileNotFoundError:
        return ""
    except Exception as e:
        # print(f"DEBUG: Error parsing LaTeX file {file_path}: {e}") 
        # import traceback
        # traceback.print_exc() 
        return ""

if __name__ == '__main__':
    # ... (rest of __main__ from previous version, no changes needed here for the test fixes) ...
    print(f"--- Processing all .tex files in: {config.RAW_LATEX_DIR} ---")
    
    if not os.path.exists(config.RAW_LATEX_DIR):
        print(f"Directory not found: {config.RAW_LATEX_DIR}. Please create it and add .tex files.")
    elif not os.listdir(config.RAW_LATEX_DIR) and not any(f.endswith(".tex") for f in os.listdir(config.RAW_LATEX_DIR)):
        dummy_tex_filename_main = "main_sample.tex"
        dummy_tex_file_main = os.path.join(config.RAW_LATEX_DIR, dummy_tex_filename_main)
        if not os.path.exists(dummy_tex_file_main): # Create only if RAW_LATEX_DIR was truly empty of .tex files
            with open(dummy_tex_file_main, "w", encoding="utf-8") as f:
                f.write("\\documentclass{article}\n")
                f.write("\\title{Sample for Main Execution}\n")
                f.write("\\begin{document}\n")
                f.write("\\section{Test Section in Main}\n")
                f.write("Content for main execution test $1+1=2$.\n")
                f.write("\\end{document}\n")
            print(f"Created dummy file for main execution: {dummy_tex_file_main}")
    
    actual_files_found = any(f.endswith(".tex") for f in os.listdir(config.RAW_LATEX_DIR)) if os.path.exists(config.RAW_LATEX_DIR) else False
    if not actual_files_found:
         print(f"No .tex files found in {config.RAW_LATEX_DIR} to process for __main__ demo.")
         exit()

    processed_count = 0
    for filename in os.listdir(config.RAW_LATEX_DIR):
        if filename.endswith(".tex"):
            file_path = os.path.join(config.RAW_LATEX_DIR, filename)
            print(f"\n--- Parsing: {filename} ---")
            parsed_content = parse_latex_file(file_path)
            
            if parsed_content:
                print(f"--- Parsed Content from {filename} (first 500 chars) ---")
                print(parsed_content[:500] + "..." if len(parsed_content) > 500 else parsed_content)
                
                output_filename = os.path.splitext(filename)[0] + "_parsed.txt"
                output_path = os.path.join(config.PARSED_LATEX_OUTPUT_DIR, output_filename)
                
                if not os.path.exists(config.PARSED_LATEX_OUTPUT_DIR):
                    os.makedirs(config.PARSED_LATEX_OUTPUT_DIR)
                    
                with open(output_path, "w", encoding="utf-8") as f_out:
                    f_out.write(parsed_content)
                print(f"Saved parsed LaTeX content to: {output_path}")
                processed_count += 1
            else:
                print(f"Could not parse or empty content for: {filename}")
        # else: # Only process .tex files
            # print(f"Skipping non-.tex file: {filename}")
    print(f"\n--- Finished processing. Parsed {processed_count} .tex files. ---")