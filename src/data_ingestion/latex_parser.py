import os
import re
from pylatexenc.latexwalker import LatexWalker, LatexCharsNode, LatexCommentNode, LatexGroupNode, LatexMacroNode, LatexMathNode, LatexEnvironmentNode
from pylatexenc.latex2text import LatexNodes2Text 

from src import config

def custom_latex_to_text(latex_str: str, preserve_section_commands=False) -> str:
    """
    Converts LaTeX string to a cleaner text representation,
    attempting to preserve math and structure.
    
    Args:
        latex_str (str): The LaTeX string to convert
        preserve_section_commands (bool): Whether to preserve original section commands 
                                         instead of transforming them. Defaults to False.
    """
    if not latex_str.strip(): # Handle empty input string
        return ""

    lw = LatexWalker(latex_str)
    nodelist, _, _ = lw.get_latex_nodes()
    
    if nodelist is None: 
        nodelist = []


    class CustomLatexNodes2Text(LatexNodes2Text):
        def __init__(self, preserve_section_commands=False, **kwargs):
            super().__init__(math_mode='verbatim', **kwargs)
            self.preserve_section_commands = preserve_section_commands

        def _format_structural_macro(self, node, macroname, conversion_state):
            arg_text = ""
            if node.nodeargs and node.nodeargs[0] and hasattr(node.nodeargs[0], 'nodelist'): 
                arg_text = self.nodelist_to_text(node.nodeargs[0].nodelist, conversion_state=conversion_state)
            
            if self.preserve_section_commands:
                return f"\n\n\\{macroname}{{{arg_text}}}\n"
            else:
                # Original transformation behavior
                if macroname == 'section':
                    return f"\n\n§ {arg_text.upper()}\n"
                elif macroname == 'subsection':
                    return f"\n\n§.§ {arg_text}\n"
                elif macroname == 'subsubsection':
                    return f"\n\n§.§.§ {arg_text}\n"
                elif macroname == 'paragraph':
                    return f"\n\n{arg_text}\n"
                else:
                    return f"\n\n{arg_text}\n"

        # Keep these std handlers as they are good practice, 
        # even if we also add a direct check in convert_node.
        def latex_std_section(self, node, conversion_state, **kwargs):
            return self._format_structural_macro(node, 'section', conversion_state)

        def latex_std_subsection(self, node, conversion_state, **kwargs):
            return self._format_structural_macro(node, 'subsection', conversion_state)

        def latex_std_subsubsection(self, node, conversion_state, **kwargs):
            return self._format_structural_macro(node, 'subsubsection', conversion_state)

        def latex_std_paragraph(self, node, conversion_state, **kwargs): 
            return self._format_structural_macro(node, 'paragraph', conversion_state)
        
        def convert_node(self, node, conversion_state=None):
            if node is None:
                return ""
            
            if node.isNodeType(LatexCharsNode):
                return node.chars

            if node.isNodeType(LatexMathNode):
                return node.latex_verbatim() 

            if node.isNodeType(LatexMacroNode):
                # --- BEGIN ADDED/MODIFIED SECTION FOR STRUCTURAL MACROS ---
                # Explicitly catch and handle structural macros first.
                if node.macroname in ['section', 'subsection', 'subsubsection', 'paragraph']:
                    return self._format_structural_macro(node, node.macroname, conversion_state)
                # --- END ADDED/MODIFIED SECTION FOR STRUCTURAL MACROS ---

                # Original logic for other macros:
                if node.macroname in ['textit', 'textbf', 'emph']: 
                    if node.nodeargs and node.nodeargs[0] and hasattr(node.nodeargs[0], 'nodelist'):
                        return self.nodelist_to_text(node.nodeargs[0].nodelist, conversion_state=conversion_state)
                    return ""
                
                # Macros to remove entirely
                if node.macroname in ['documentclass', 'usepackage', 
                                      'title', 'author', 'date', 'maketitle',
                                      'label', 'ref', 'cite', 'pagestyle', 'thispagestyle',
                                      'includegraphics', 'figureheight', 'figurewidth', 'caption',
                                      'tableofcontents', 'listoffigures', 'listoftables',
                                      'appendix', 'bibliographystyle', 'bibliography',
                                      ]:
                    return "" 
                
                # For other unhandled macros, try to extract text from their arguments
                text_from_args = ""
                if hasattr(node, 'nodeargs') and node.nodeargs:
                    for arg_node in node.nodeargs:
                        if hasattr(arg_node, 'nodelist') and arg_node.nodelist:
                            text_from_args += self.nodelist_to_text(arg_node.nodelist, conversion_state=conversion_state)
                
                if text_from_args:
                    return text_from_args
                
                # Fallback for macros not explicitly handled above
                return super().convert_node(node, conversion_state=conversion_state)


            if node.isNodeType(LatexCommentNode):
                return "" 
            
            if node.isNodeType(LatexEnvironmentNode):
                if node.environmentname in ['document', 'abstract', 'center', 'figure', 'table']:
                     return self.nodelist_to_text(node.nodelist, conversion_state=conversion_state)
                return self.nodelist_to_text(node.nodelist, conversion_state=conversion_state)


            if node.isNodeType(LatexGroupNode):
                return self.nodelist_to_text(node.nodelist, conversion_state=conversion_state)

            return super().convert_node(node, conversion_state=conversion_state)


    converter = CustomLatexNodes2Text(preserve_section_commands=preserve_section_commands)
    text_content = converter.nodelist_to_text(nodelist)
    
    # Final cleanup of excessive newlines that might result from preserved structural commands
    text_content = re.sub(r'(\n\s*){3,}', '\n\n', text_content) 
    text_content = text_content.strip()
    return text_content


def parse_latex_file(file_path, preserve_section_commands=False):
    """
    Parse a LaTeX file and convert it to plain text
    
    Args:
        file_path (str): Path to the LaTeX file
        preserve_section_commands (bool, optional): Whether to preserve original section commands 
                                                   instead of transforming them. Defaults to False.
    
    Returns:
        str: The parsed text content
    """
    try:
        # Read the file
        with open(file_path, "r", encoding="utf-8") as f:
            latex_content = f.read()
        
        # Use the modified converter function with the new option
        parsed_text = custom_latex_to_text(latex_content, preserve_section_commands)
        return parsed_text
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return ""
    except Exception as e:
        print(f"Error parsing LaTeX file {file_path}: {e}")
        return ""

if __name__ == '__main__':
    print(f"--- Processing all .tex files in: {config.RAW_LATEX_DIR} ---")
    
    if not os.path.exists(config.RAW_LATEX_DIR):
        print(f"Directory not found: {config.RAW_LATEX_DIR}. Please create it and add .tex files.")
    elif not os.listdir(config.RAW_LATEX_DIR):
        # Create a dummy file for demonstration if the directory is empty
        dummy_tex_filename_main = "main_sample.tex"
        dummy_tex_file_main = os.path.join(config.RAW_LATEX_DIR, dummy_tex_filename_main)
        if not os.path.exists(dummy_tex_file_main):
            with open(dummy_tex_file_main, "w", encoding="utf-8") as f:
                f.write("\\documentclass{article}\n")
                f.write("\\title{Sample for Main Execution}\n")
                f.write("\\begin{document}\n")
                f.write("\\section{Test Section in Main}\n")
                f.write("Content for main execution test $1+1=2$.\n")
                f.write("\\end{document}\n")
            print(f"Created dummy file for main execution: {dummy_tex_file_main}")
        # Fallthrough to process this newly created file or any existing ones.
    
    # Check again if directory has files after potential dummy file creation
    if not os.listdir(config.RAW_LATEX_DIR):
         print(f"Still no files found in {config.RAW_LATEX_DIR} after attempting to create a dummy file.")

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
        else:
            print(f"Skipping non-.tex file: {filename}")
    print(f"\n--- Finished processing. Parsed {processed_count} .tex files. ---")