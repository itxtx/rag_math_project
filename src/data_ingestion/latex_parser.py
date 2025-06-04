# src/data_ingestion/latex_parser.py
import os
import re
import sys 
from pylatexenc.latexwalker import LatexWalker, LatexCharsNode, LatexCommentNode, LatexGroupNode, LatexMacroNode, LatexMathNode, LatexEnvironmentNode, get_default_latex_context_db
from pylatexenc.latex2text import LatexNodes2Text, MacroTextSpec 
from pylatexenc.macrospec import MacroSpec, MacroStandardArgsParser, LatexContextDb

try:
    from src import config
except ModuleNotFoundError:
    # This allows the script to be imported by other modules in src
    # without failing if config is not immediately on path during initial load.
    # The __main__ block will handle path adjustment for direct execution.
    config = None # Placeholder
import latex_processor # <<< NEW IMPORT


def custom_latex_to_text(latex_str: str) -> str: # This function now receives preprocessed latex_str
    print(f"DEBUG custom_latex_to_text: Input LaTeX string (first 200 chars *after* latex_processor): '{latex_str[:200]}...'")
    if not latex_str.strip():
        print("DEBUG custom_latex_to_text: LaTeX string is empty or whitespace.")
        return ""
    
    original_recursion_limit = sys.getrecursionlimit()
    increased_limit = max(original_recursion_limit, 4000) 
    nodelist = []
    parsing_error_occurred = False
    try:
        if original_recursion_limit < increased_limit:
            sys.setrecursionlimit(increased_limit) 
            print(f"DEBUG custom_latex_to_text: Temporarily increased recursion limit to {sys.getrecursionlimit()}")
        
        lw = LatexWalker(latex_str) 
        parsing_result = lw.get_latex_nodes() 

        if parsing_result is None: nodelist = []
        else: nodelist, _, _ = parsing_result; nodelist = nodelist or []
            
    except RecursionError as re_err_inner:
        print(f"ERROR (inner): Recursion depth exceeded during LatexWalker.get_latex_nodes() for content starting with '{latex_str[:50]}...'. Error: {re_err_inner}")
        parsing_error_occurred = True
        return "" 
    except Exception as e_walker:
        print(f"ERROR (inner): Exception during LatexWalker.get_latex_nodes() for content starting with '{latex_str[:50]}...'. Error: {e_walker}")
        parsing_error_occurred = True
        return ""
    finally:
        if sys.getrecursionlimit() != original_recursion_limit:
            sys.setrecursionlimit(original_recursion_limit)
            print(f"DEBUG custom_latex_to_text: Reset recursion limit to {original_recursion_limit}")

    if parsing_error_occurred: return ""

    print(f"DEBUG custom_latex_to_text: Initial nodelist length: {len(nodelist)}")
    if nodelist: print(f"DEBUG custom_latex_to_text: First node type: {type(nodelist[0]).__name__}")

    class CustomLatexNodes2Text(LatexNodes2Text):
        def __init__(self, **kwargs):
            super().__init__(math_mode="verbatim", **kwargs) 
            
            if not hasattr(self, 'latex_context_db') or self.latex_context_db is None:
                print("WARN: self.latex_context_db not set by superclass or is None. Initializing a new default one.")
                self.latex_context_db = get_default_latex_context_db()
            
            print(f"DEBUG __init__: CustomLatexNodes2Text initialized.")

            # MacroTextSpec for commands that latex_processor might not fully expand
            # or for which pylatexenc's argument parsing is beneficial.
            custom_arg_macros_text_specs = []
            custom_arg_macros_text_specs.extend([
                MacroTextSpec("Zn", simplify_repl=r"\Z_{%(1)s}"),          
                MacroTextSpec("Znx", simplify_repl=r"\Z_{%(1)s}^\times"), 
                MacroTextSpec("fieldext", simplify_repl=r"%(1)s/%(2)s"),    
                MacroTextSpec("degree", simplify_repl=r"[%(1)s:%(2)s]"),    
                MacroTextSpec("ideal", simplify_repl=r"\langle %(1)s \rangle"), 
                MacroTextSpec("abs", simplify_repl=r"\left| %(1)s \right|"),  
                MacroTextSpec("norm", simplify_repl=r"\text{N}(%(1)s)"), # As per user's latest definition
                MacroTextSpec("diff", simplify_repl=r"\frac{d%(1)s}{d%(2)s}"), 
                MacroTextSpec("pdiff", simplify_repl=r"\frac{\partial %(1)s}{\partial %(2)s}"),
                MacroTextSpec("bvec", simplify_repl=r"\bm{%(1)s}"),         
                MacroTextSpec("powerset", simplify_repl=r"\mathcal{P}(%(1)s)"), 
                MacroTextSpec("dual", simplify_repl=r"%(1)s^*"),
                MacroTextSpec("Lp", simplify_repl=r"L^{%(1)s}"),
                MacroTextSpec("lp", simplify_repl=r"\ell^{%(1)s}"),
                MacroTextSpec("linop", simplify_repl=r"\mathcal{L}(%(1)s, %(2)s)"),
                MacroTextSpec("boundedlinop", simplify_repl=r"\mathcal{L}(%(1)s)"),
                MacroTextSpec("inner", simplify_repl=r"\langle %(1)s, %(2)s \rangle"),
                MacroTextSpec("conj", simplify_repl=r"\overline{%(1)s}"),
                MacroTextSpec("herm", simplify_repl=r"%(1)s^H"),
                MacroTextSpec("vec", simplify_repl=r"\mathbf{%(1)s}") 
            ])

            if custom_arg_macros_text_specs:
                category_name = 'custom_argument_macros_text_cat_v5' # New version
                self.latex_context_db.add_context_category(
                    category_name, 
                    macros=custom_arg_macros_text_specs,
                    prepend=True 
                )
                print(f"DEBUG __init__: Category '{category_name}' now has {len(custom_arg_macros_text_specs)} MacroTextSpecs.")


        def convert_node(self, node):
            # ... (convert_node logic remains the same) ...
            # The `defining_macros_to_remove` list should still contain \newcommand, \DeclareMathOperator etc.
            # because latex_processor.remove_command_definitions aims to remove these lines.
            # If any remnants are parsed as macros, this will strip them.
            if node is None: return ""
            if node.isNodeType(LatexCharsNode): return node.chars
            if node.isNodeType(LatexMathNode): return node.latex_verbatim()
            if node.isNodeType(LatexMacroNode):
                print(f"DEBUG convert_node: Encountered LatexMacroNode with macroname: '{node.macroname}'")
                if node.macroname in ['textit', 'textbf', 'emph']: 
                    if node.nodeargs and len(node.nodeargs) > 0 and \
                       node.nodeargs[0] is not None and hasattr(node.nodeargs[0], 'nodelist'):
                        return self.nodelist_to_text(node.nodeargs[0].nodelist)
                    return "" 
                defining_macros_to_remove = [
                    'title', 'author', 'date', 'maketitle', 'documentclass', 'usepackage', 
                    'label', 'ref', 'cite', 'pagestyle', 'thispagestyle', 'includegraphics', 
                    'figureheight', 'figurewidth', 'caption', 'tableofcontents', 'listoffigures', 
                    'listoftables', 'appendix', 'bibliographystyle', 'bibliography', 'hspace', 
                    'vspace', 'hfill', 'vfill', 'centering', 'noindent', 'indent', 'newpage', 
                    'clearpage', 'linebreak', 'nolinebreak', 'setlength', 'addtolength', 
                    'setcounter', 'addtocounter', 'selectfont', 'item', 'newtheorem', 
                    'newenvironment', 'renewenvironment', 'newcommand', 'renewcommand', 
                    'providecommand', 'DeclareMathOperator', 'DeclareRobustCommand', 
                    'geometry', 'setstretch', 'linespread', 'hyphenation', 'url', 'href', 
                    'graphicspath', 'input', 'include', 'RequirePackage', 'LoadClass', 
                    'ProcessOptions', 'PassOptionsToPackage', 'newif', 'ifdefined', 'ifx', 
                    'fi', 'else', 'let', 'def', 'edef', 'gdef', 'xdef',
                    'AtBeginDocument', 'AtEndDocument', 'AtEndOfClass', 'AtEndOfPackage',
                    'ExplSyntaxOn', 'ExplSyntaxOff', 'ProvidesPackage', 'NeedsTeXFormat'
                ]
                if node.macroname in defining_macros_to_remove:
                    print(f"DEBUG convert_node: Removing defining/layout macro '{node.macroname}' by returning empty string.")
                    return "" 
                print(f"DEBUG convert_node: For macro '{node.macroname}', calling super().convert_node() for dispatch.")
                return super().convert_node(node)
            if node.isNodeType(LatexCommentNode): return ""
            if node.isNodeType(LatexEnvironmentNode):
                if node.environmentname == 'document':
                    print("DEBUG convert_node: Processing 'document' environment content.")
                    return self.nodelist_to_text(node.nodelist)
                if node.environmentname in ['itemize', 'enumerate', 'description']:
                    print(f"DEBUG convert_node: Letting super handle environment '{node.environmentname}'.")
                    return super().convert_node(node)
                environments_to_process_content = [
                    'abstract', 'center', 'figure', 'table', 'theorem', 'lemma', 'proof', 
                    'definition', 'corollary', 'example', 'remark'
                ]
                if node.environmentname in environments_to_process_content:
                    print(f"DEBUG convert_node: Processing content of environment '{node.environmentname}'.")
                    return self.nodelist_to_text(node.nodelist) 
                environments_to_remove = ['comment']
                if node.environmentname in environments_to_remove:
                    print(f"DEBUG convert_node: Removing environment '{node.environmentname}'.")
                    return ""
                print(f"DEBUG convert_node: Environment '{node.environmentname}' not explicitly handled, calling super.")
                return super().convert_node(node)
            if node.isNodeType(LatexGroupNode): return self.nodelist_to_text(node.nodelist)
            return super().convert_node(node)

    document_node = None
    if nodelist:
        for i, node in enumerate(nodelist):
            if node.isNodeType(LatexEnvironmentNode) and node.environmentname == 'document':
                document_node = node
                print(f"DEBUG custom_latex_to_text: Found 'document' environment at top-level node index {i}.")
                break
    
    converter = CustomLatexNodes2Text()
    text_content = "" 

    if document_node:
        print("DEBUG custom_latex_to_text: Processing ONLY content of 'document' environment.")
        text_content = converter.nodelist_to_text(document_node.nodelist)
    else:
        print("WARNING custom_latex_to_text: No top-level 'document' environment found. Will process full nodelist.")
        text_content = converter.nodelist_to_text(nodelist) 
    
    if text_content: 
        text_content = text_content.replace('\r\n', '\n').replace('\r', '\n')
        lines = text_content.split('\n')
        processed_lines = []
        for line in lines:
            stripped_line = line.strip()
            list_marker_match = re.match(r"^(\s*)([\*\-]\s*|\d+\.\s*)(.*)", stripped_line)
            if list_marker_match:
                marker = list_marker_match.group(2).strip(); item_content = list_marker_match.group(3).strip()
                if marker.endswith('.'): processed_lines.append(f"{marker} {item_content}")
                else: processed_lines.append(f"{marker} {item_content}")
            elif stripped_line: processed_lines.append(stripped_line)
        text_content = "\n".join(processed_lines)
        text_content = re.sub(r'\n{3,}', '\n\n', text_content) 
        text_content = text_content.strip() 
    
    print(f"DEBUG custom_latex_to_text: Final text content (first 200 chars): '{text_content[:200]}...'")
    return text_content


def parse_latex_file(file_path: str) -> str:
    print(f"Attempting to parse LaTeX file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            latex_content_original = f.read()
        
        if not latex_content_original.strip():
            print(f"LaTeX file is empty or contains only whitespace: {file_path}")
            return ""
        
        print("DEBUG parse_latex_file: Starting custom command expansion with new latex_processor module...")
        expanded_content = latex_processor.process_latex_document(latex_content_original)
        
        print("DEBUG parse_latex_file: Removing command definitions with new latex_processor module...")
        content_for_pylatexenc = latex_processor.remove_command_definitions(expanded_content)
        
        if content_for_pylatexenc != latex_content_original: 
            print("DEBUG parse_latex_file: Content was modified by new latex_processor.")
        else:
            print("DEBUG parse_latex_file: Content was NOT modified by new latex_processor.")
            
        text_representation = custom_latex_to_text(content_for_pylatexenc) 
        
        if not text_representation.strip() and latex_content_original.strip():
            print(f"INFO: custom_latex_to_text returned empty/whitespace for non-empty file: {file_path}")
            
        return text_representation

    except RecursionError as re_err: 
        print(f"ERROR: Recursion depth exceeded while parsing LaTeX file {file_path}. Error: {re_err}")
        return "" 
    except FileNotFoundError:
        print(f"ERROR: LaTeX file not found: {file_path}")
        return ""
    except Exception as e:
        print(f"DEBUG: Error parsing LaTeX file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return ""

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) # Assuming src/data_ingestion/latex_parser.py
    if project_root not in sys.path:
        sys.path.insert(0, project_root)    
    # ... (if __name__ == '__main__' block remains the same) ...
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
    if not os.path.exists(raw_latex_dir_to_use): exit()
    if not os.path.exists(parsed_latex_output_dir_to_use): os.makedirs(parsed_latex_output_dir_to_use)
    tex_files_in_dir = [f for f in os.listdir(raw_latex_dir_to_use) if f.endswith(".tex")]
    if not tex_files_in_dir:
        pass # dummy file creation
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
            with open(output_path, "w", encoding="utf-8") as f_out: f_out.write(parsed_content)
            print(f"Saved parsed LaTeX content to: {output_path}")
            processed_count += 1
        else: print(f"Could not parse or empty content for: {filename}")
    if processed_count == 0 and tex_files_in_dir: print("\nWARNING: No .tex files were successfully parsed.")
    elif not tex_files_in_dir: print(f"\nNo .tex files found in {raw_latex_dir_to_use} for demo.")
    print(f"\n--- Finished processing. Parsed {processed_count} .tex files. ---")
