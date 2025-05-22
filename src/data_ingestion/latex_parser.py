# src/data_ingestion/latex_parser.py
import os
import re
import sys # For recursion limit, though not changing it globally by default
from pylatexenc.latexwalker import LatexWalker, LatexCharsNode, LatexCommentNode, LatexGroupNode, LatexMacroNode, LatexMathNode, LatexEnvironmentNode, get_default_latex_context_db
from pylatexenc.latex2text import LatexNodes2Text, MacroTextSpec 
from pylatexenc.macrospec import MacroSpec, MacroStandardArgsParser, LatexContextDb

from src import config

def _preprocess_custom_latex_commands(latex_content: str) -> str:
    """
    Preprocesses LaTeX content to expand simple custom command definitions
    (\DeclareMathOperator and simple \newcommand/\renewcommand without arguments 
    in their definition) before main parsing.
    """
    print("DEBUG _preprocess_custom_latex_commands: Starting preprocessing...")
    processed_content = latex_content
    custom_commands_to_replace = {} 

    # \DeclareMathOperator{\cmd}{replacement_text}
    declaremath_pattern = re.compile(r"\\DeclareMathOperator\s*\{\s*\\(\w+)\s*\}\s*\{(.*?)\}")
    for match in declaremath_pattern.finditer(latex_content):
        cmd_name = match.group(1)
        replacement = match.group(2).strip() 
        custom_commands_to_replace[f"\\{cmd_name}"] = replacement
        print(f"DEBUG _preprocess: Found DeclareMathOperator: \\{cmd_name} -> {replacement}")

    # Simple \newcommand{\cmd}{replacement_text} (no arguments in definition part)
    newcommand_simple_pattern = re.compile(r"\\newcommand\s*\{\s*\\(\w+)\s*\}\s*\{(.*?)\}(?!\s*\[)")
    for match in newcommand_simple_pattern.finditer(latex_content):
        cmd_name = match.group(1)
        replacement = match.group(2).strip()
        if f"\\{cmd_name}" not in custom_commands_to_replace:
            custom_commands_to_replace[f"\\{cmd_name}"] = replacement
            print(f"DEBUG _preprocess: Found simple newcommand: \\{cmd_name} -> {replacement}")
        else:
            print(f"DEBUG _preprocess: newcommand for \\{cmd_name} skipped (already defined).")
    
    # Simple \renewcommand{\cmd}{replacement_text} (no arguments in definition part)
    renewcommand_simple_pattern = re.compile(r"\\renewcommand\s*\{\s*\\(\w+)\s*\}\s*\{(.*?)\}(?!\s*\[)")
    for match in renewcommand_simple_pattern.finditer(latex_content):
        cmd_name = match.group(1)
        replacement = match.group(2).strip()
        custom_commands_to_replace[f"\\{cmd_name}"] = replacement 
        print(f"DEBUG _preprocess: Found simple renewcommand: \\{cmd_name} -> {replacement}")

    manual_operatorname_defs = {
        '\\kerphi': r'\operatorname{ker}\phi',
        '\\imphi': r'\operatorname{im}\phi',
        '\\Gal': r'\operatorname{Gal}',
        '\\Aut': r'\operatorname{Aut}',
        '\\degpoly': r'\operatorname{deg}'
    }
    for cmd, definition in manual_operatorname_defs.items():
        if cmd in latex_content and definition.split('{')[0] in latex_content: 
             # For these manually defined replacements, ensure they are already re.sub safe
             # or make them so. Here, they contain \ which needs to be \\ for re.sub's repl.
             custom_commands_to_replace[cmd] = definition.replace('\\', '\\\\') 
             print(f"DEBUG _preprocess: Added manual operatorname-like cmd: {cmd} -> {custom_commands_to_replace[cmd]}")


    if custom_commands_to_replace:
        print(f"DEBUG _preprocess: Applying {len(custom_commands_to_replace)} simple command pre-replacements.")
        sorted_cmd_keys = sorted(custom_commands_to_replace.keys(), key=len, reverse=True)
        
        for cmd_key in sorted_cmd_keys:
            replacement_text = custom_commands_to_replace[cmd_key]
            
            # Ensure all backslashes in replacement_text are escaped for re.sub,
            # unless it's already correctly escaped (e.g., from manual_operatorname_defs).
            # The manual ones are already made `\\\\` for `\`.
            # For those from regex (like `\mathbb{R}`), they are single `\`.
            # So, if it's not from manual_operatorname_defs (which are already `\\\\`), then escape.
            # A simpler, more general approach: if it's not already `\\\\`, make it `\\\\`.
            # This might be too aggressive if `replacement_text` is just "Ker".
            
            # Refined logic: if replacement_text was from manual_operatorname_defs, it's already `\\\\`.
            # Otherwise, it's from regex and needs `\` -> `\\\\`.
            if cmd_key not in manual_operatorname_defs: # If derived from regex and contains '\'
                final_replacement_text = replacement_text.replace('\\', '\\\\')
            else: # Already correctly escaped for re.sub if from manual_operatorname_defs
                final_replacement_text = replacement_text
            
            pattern_to_replace = r"(" + re.escape(cmd_key) + r")(?![a-zA-Z])" 
            
            # print(f"DEBUG _preprocess: Replacing '{cmd_key}' with '{final_replacement_text}'")
            try:
                processed_content = re.sub(pattern_to_replace, final_replacement_text, processed_content)
            except re.error as e_re:
                print(f"ERROR during re.sub for cmd_key='{cmd_key}', replacement='{final_replacement_text}': {e_re}")
                # Skip this problematic replacement and continue
                continue
        print("DEBUG _preprocess_custom_latex_commands: Preprocessing for simple commands complete.")
    else:
        print("DEBUG _preprocess_custom_latex_commands: No simple custom commands found for preprocessing.")
        
    return processed_content


def custom_latex_to_text(latex_str: str) -> str:
    print(f"DEBUG custom_latex_to_text: Input LaTeX string (first 200 chars after potential preprocessing): '{latex_str[:200]}...'")
    if not latex_str.strip():
        print("DEBUG custom_latex_to_text: LaTeX string is empty or whitespace.")
        return ""
    
    # --- Try increasing recursion limit locally for LatexWalker if needed ---
    # original_recursion_limit = sys.getrecursionlimit()
    # try:
    #     if original_recursion_limit < 3000: # Example threshold
    #         sys.setrecursionlimit(3000) 
    #         print(f"DEBUG custom_latex_to_text: Temporarily increased recursion limit to {sys.getrecursionlimit()}")
    # except RuntimeError: # Some environments might restrict this
    #     print("WARN: Could not increase recursion limit.")
    # --- End of recursion limit adjustment ---

    lw = LatexWalker(latex_str) 
    parsing_result = lw.get_latex_nodes() 

    # if original_recursion_limit < sys.getrecursionlimit(): # Reset if changed
    #     sys.setrecursionlimit(original_recursion_limit)
    #     print(f"DEBUG custom_latex_to_text: Reset recursion limit to {original_recursion_limit}")


    if parsing_result is None: nodelist = []
    else: nodelist, _, _ = parsing_result; nodelist = nodelist or []
    
    print(f"DEBUG custom_latex_to_text: Initial nodelist length: {len(nodelist)}")
    if nodelist: print(f"DEBUG custom_latex_to_text: First node type: {type(nodelist[0]).__name__}")

    class CustomLatexNodes2Text(LatexNodes2Text):
        def __init__(self, **kwargs):
            super().__init__(math_mode="verbatim", **kwargs) 
            
            if not hasattr(self, 'latex_context_db') or self.latex_context_db is None:
                print("WARN: self.latex_context_db not set by superclass or is None. Initializing a new default one.")
                self.latex_context_db = get_default_latex_context_db()
            
            print(f"DEBUG __init__: CustomLatexNodes2Text initialized.")

            custom_arg_macros_text_specs = []
            custom_arg_macros_text_specs.extend([
                MacroTextSpec("Zn", simplify_repl=r"\Z_{%(1)s}"),          
                MacroTextSpec("Znx", simplify_repl=r"\Z_{%(1)s}^\times"), 
                MacroTextSpec("fieldext", simplify_repl=r"%(1)s/%(2)s"),    
                MacroTextSpec("degree", simplify_repl=r"[%(1)s:%(2)s]"),    
                MacroTextSpec("ideal", simplify_repl=r"\langle %(1)s \rangle"), 
                MacroTextSpec("abs", simplify_repl=r"\left| %(1)s \right|"),  
                MacroTextSpec("norm", simplify_repl=r"\text{N}(%(1)s)"), 
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
                category_name = 'custom_argument_macros_text_cat_v4' 
                self.latex_context_db.add_context_category(
                    category_name, 
                    macros=custom_arg_macros_text_specs,
                    prepend=True 
                )
                print(f"DEBUG __init__: Category '{category_name}' (prepended) now has {len(custom_arg_macros_text_specs)} MacroTextSpecs for argument-taking commands.")


        def convert_node(self, node):
            # ... (convert_node logic remains the same) ...
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
                    return self.nodelist_to_text(node.nodelist) # Removed extra newlines
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
    """
    Parses a LaTeX file: preprocesses custom commands, then extracts textual representation.
    """
    print(f"Attempting to parse LaTeX file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            latex_content_original = f.read()
        
        if not latex_content_original.strip():
            print(f"LaTeX file is empty or contains only whitespace: {file_path}")
            return ""
        
        print("DEBUG parse_latex_file: Starting custom command preprocessing...")
        latex_content_preprocessed = _preprocess_custom_latex_commands(latex_content_original) # RE-ENABLED
        if latex_content_preprocessed != latex_content_original: 
            print("DEBUG parse_latex_file: Content was modified by preprocessor.")
        else:
            print("DEBUG parse_latex_file: Content was NOT modified by preprocessor.")
            
        text_representation = custom_latex_to_text(latex_content_preprocessed)
        
        if not text_representation.strip() and latex_content_original.strip():
            print(f"INFO: custom_latex_to_text returned empty/whitespace for non-empty file: {file_path}")
            
        return text_representation

    except RecursionError as re_err: # Specifically catch RecursionError
        print(f"ERROR: Recursion depth exceeded while parsing LaTeX file {file_path}. This file is too complex for the current parser limits or has problematic syntax. Error: {re_err}")
        import traceback
        traceback.print_exc()
        return "" # Return empty for this file
    except FileNotFoundError:
        print(f"ERROR: LaTeX file not found: {file_path}")
        return ""
    except Exception as e:
        print(f"DEBUG: Error parsing LaTeX file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return ""

if __name__ == '__main__':
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
        # ... (dummy file creation) ...
        pass
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
