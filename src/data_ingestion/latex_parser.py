import os
import re
import sys
from pylatexenc.latexwalker import LatexWalker, LatexCharsNode, LatexCommentNode, LatexGroupNode, LatexMacroNode, LatexMathNode, LatexEnvironmentNode, get_default_latex_context_db
from pylatexenc.latex2text import LatexNodes2Text, MacroTextSpec
from pylatexenc.macrospec import MacroSpec, MacroStandardArgsParser, LatexContextDb # Not directly used, but good for context
from typing import Dict # Added for type hinting

# Attempt to import config, but allow script to run standalone for testing
try:
    from src import config
except ImportError:
    config = None # Will be handled in __main__

# --- Configuration for Recursion Limit ---
TARGET_RECURSION_LIMIT = 15000

# --- Set of defining/layout macros to remove during text conversion ---
DEFINING_MACROS_TO_REMOVE = {
    'title', 'author', 'date', 'maketitle', 'documentclass', 'usepackage',
    'label', 'ref', 'cite', 'pagestyle', 'thispagestyle', 'includegraphics',
    'figureheight', 'figurewidth', 'caption', 'tableofcontents', 'listoffigures',
    'listoftables', 'appendix', 'bibliographystyle', 'bibliography', 'hspace',
    'vspace', 'hfill', 'vfill', 'centering', 'noindent', 'indent', 'newpage',
    'clearpage', 'linebreak', 'nolinebreak', 'setlength', 'addtolength',
    'setcounter', 'addtocounter', 'selectfont', 'item',
    'newtheorem', 'newenvironment', 'renewenvironment', 'newcommand', 'renewcommand',
    'providecommand', 'DeclareMathOperator', 'DeclareRobustCommand',
    'geometry', 'setstretch', 'linespread', 'hyphenation', 'url', 'href',
    'graphicspath', 'input', 'include', 'RequirePackage', 'LoadClass',
    'ProcessOptions', 'PassOptionsToPackage', 'newif', 'ifdefined', 'ifx',
    'fi', 'else', 'let', 'def', 'edef', 'gdef', 'xdef',
    'AtBeginDocument', 'AtEndDocument', 'AtEndOfClass', 'AtEndOfPackage',
    'ExplSyntaxOn', 'ExplSyntaxOff', 'ProvidesPackage', 'NeedsTeXFormat'
}

def _preprocess_custom_latex_commands(latex_content: str) -> str:
    """
    Preprocesses LaTeX content to expand simple custom command definitions
    (\DeclareMathOperator and simple \newcommand/\renewcommand without arguments
    in their definition) before main parsing.
    Skips preprocessing for newcommand/renewcommand if captured definition has unbalanced braces.
    """
    print("DEBUG _preprocess_custom_latex_commands: Starting preprocessing...")
    processed_content = latex_content
    custom_commands_to_replace: Dict[str, str] = {}

    # \DeclareMathOperator{\cmd}{replacement_text}
    declaremath_pattern = re.compile(r"\\DeclareMathOperator\s*\{\s*\\(\w+)\s*\}\s*\{(.*?)\}")
    for match in declaremath_pattern.finditer(latex_content):
        cmd_name = match.group(1)
        replacement_text = match.group(2).strip()
        # If replacement is just text (like "sin"), wrap it with \operatorname
        # to ensure it's treated as a math operator.
        # If it already contains commands, trust it.
        if re.match(r"^[a-zA-Z]+$", replacement_text):
            replacement = f"\\operatorname{{{replacement_text}}}"
        else:
            replacement = replacement_text # Assume it's already valid LaTeX like \mathrm{Hom}

        custom_commands_to_replace[f"\\{cmd_name}"] = replacement
        print(f"DEBUG _preprocess: Found DeclareMathOperator: \\{cmd_name} -> {replacement}")

    # Simple \newcommand{\cmd}{replacement_text} (no arguments in definition part)
    newcommand_simple_pattern = re.compile(r"\\newcommand\s*\{\s*\\(\w+)\s*\}\s*\{(.*?)\}(?!\s*\[)")
    for match in newcommand_simple_pattern.finditer(latex_content):
        cmd_name = match.group(1)
        replacement = match.group(2).strip() # Definition text

        if replacement.count('{') != replacement.count('}'):
            print(f"WARN _preprocess: Captured definition for \\{cmd_name} ('{replacement}') in newcommand seems to have unbalanced braces. Skipping this pre-replacement.")
            continue

        dict_key = f"\\{cmd_name}"
        if dict_key not in custom_commands_to_replace:
            custom_commands_to_replace[dict_key] = replacement
            print(f"DEBUG _preprocess: Found simple newcommand: {dict_key} -> {replacement}")

    # Simple \renewcommand{\cmd}{replacement_text} (no arguments in definition part)
    renewcommand_simple_pattern = re.compile(r"\\renewcommand\s*\{\s*\\(\w+)\s*\}\s*\{(.*?)\}(?!\s*\[)")
    for match in renewcommand_simple_pattern.finditer(latex_content):
        cmd_name = match.group(1)
        replacement = match.group(2).strip()

        if replacement.count('{') != replacement.count('}'):
            print(f"WARN _preprocess: Captured definition for \\{cmd_name} ('{replacement}') in renewcommand seems to have unbalanced braces. Skipping this pre-replacement.")
            continue

        dict_key = f"\\{cmd_name}"
        custom_commands_to_replace[dict_key] = replacement
        print(f"DEBUG _preprocess: Found simple renewcommand: {dict_key} -> {replacement}")

    manual_defs_to_replace = {
        '\\kerphi': r'\operatorname{ker}\phi',
        '\\imphi': r'\operatorname{im}\phi',
        '\\Gal': r'\operatorname{Gal}',
        '\\Aut': r'\operatorname{Aut}',
        '\\degpoly': r'\operatorname{deg}'
    }
    for cmd, definition in manual_defs_to_replace.items():
        if cmd in latex_content:
             custom_commands_to_replace[cmd] = definition
             print(f"DEBUG _preprocess: Added/updated manual cmd: {cmd} -> {definition}")


    if custom_commands_to_replace:
        print(f"DEBUG _preprocess: Applying {len(custom_commands_to_replace)} simple command pre-replacements.")
        sorted_cmd_keys = sorted(custom_commands_to_replace.keys(), key=len, reverse=True)

        for cmd_key in sorted_cmd_keys:
            literal_replacement_text = custom_commands_to_replace[cmd_key]
            pattern_to_replace = r"(" + re.escape(cmd_key) + r")(?![a-zA-Z])"
            try:
                processed_content = re.sub(pattern_to_replace, lambda m: literal_replacement_text, processed_content)
            except re.error as e_re:
                print(f"ERROR during re.sub for cmd_key='{cmd_key}', replacement='{literal_replacement_text}': {e_re}")
                continue
        print("DEBUG _preprocess_custom_latex_commands: Preprocessing for simple commands complete.")
    else:
        print("DEBUG _preprocess_custom_latex_commands: No simple custom commands found for preprocessing.")

    return processed_content


def custom_latex_to_text(latex_str: str) -> str:
    """
    Converts a LaTeX string to a cleaner textual representation using pylatexenc.
    Handles custom commands and environments.
    """
    print(f"DEBUG custom_latex_to_text: Input LaTeX string (first 200 chars after potential preprocessing): '{latex_str[:200]}...'")
    if not latex_str.strip():
        print("DEBUG custom_latex_to_text: LaTeX string is empty or whitespace.")
        return ""

    original_recursion_limit = sys.getrecursionlimit()
    effective_recursion_limit = max(original_recursion_limit, TARGET_RECURSION_LIMIT)

    nodelist = []
    parsing_error_occurred = False
    try:
        if original_recursion_limit < effective_recursion_limit:
            sys.setrecursionlimit(effective_recursion_limit)
            print(f"DEBUG custom_latex_to_text: Temporarily increased recursion limit from {original_recursion_limit} to {sys.getrecursionlimit()}")

        latex_context = get_default_latex_context_db()
        lw = LatexWalker(latex_str, latex_context=latex_context)
        parsing_result = lw.get_latex_nodes(pos=0)

        if parsing_result is None:
             nodelist = []
        else:
            nodelist = parsing_result[0] if parsing_result[0] is not None else []

    except RecursionError as re_err_inner:
        print(f"ERROR (inner): Recursion depth exceeded ({sys.getrecursionlimit()}) during LatexWalker.get_latex_nodes() for content starting with '{latex_str[:100]}...'. Error: {re_err_inner}")
        parsing_error_occurred = True
        return ""
    except Exception as e_walker:
        print(f"ERROR (inner): Exception during LatexWalker.get_latex_nodes() for content starting with '{latex_str[:100]}...'. Error: {type(e_walker).__name__}: {e_walker}")
        import traceback
        traceback.print_exc()
        parsing_error_occurred = True
        return ""
    finally:
        if sys.getrecursionlimit() != original_recursion_limit:
            sys.setrecursionlimit(original_recursion_limit)
            print(f"DEBUG custom_latex_to_text: Reset recursion limit to {original_recursion_limit}")

    if parsing_error_occurred:
        return ""

    print(f"DEBUG custom_latex_to_text: Initial nodelist length: {len(nodelist)}")
    if nodelist:
        print(f"DEBUG custom_latex_to_text: First node type: {type(nodelist[0]).__name__ if nodelist else 'N/A'}")

    class CustomLatexNodes2Text(LatexNodes2Text):
        def __init__(self, **kwargs):
            super().__init__(math_mode="verbatim", **kwargs) # Crucial for preserving LaTeX in math

            if not hasattr(self, 'latex_context_db') or self.latex_context_db is None:
                print("WARN: CustomLatexNodes2Text: self.latex_context_db not set by superclass or is None. Initializing a new default one.")
                self.latex_context_db = get_default_latex_context_db()

            print(f"DEBUG CustomLatexNodes2Text __init__: Initialized.")

            # Base list of argument-taking macros
            custom_arg_macros_text_specs_list = [
                MacroTextSpec("Zn", simplify_repl=r"\Z_{%(1)s}"),
                MacroTextSpec("Znx", simplify_repl=r"\Z_{%(1)s}^\times"),
                MacroTextSpec("fieldext", simplify_repl=r"%(1)s/%(2)s"),
                MacroTextSpec("degree", simplify_repl=r"\[%(1)s:%(2)s]"),
                MacroTextSpec("ideal", simplify_repl=r"\langle %(1)s \rangle"),
                MacroTextSpec("abs", simplify_repl=r"\left| %(1)s \right|"),
                MacroTextSpec("norm", simplify_repl=r"\text{N}(%(1)s)"), # Or \operatorname{N}
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
            ]
            
            # Use a dictionary to manage specs and avoid duplicates, ensuring one spec per macroname
            specs_map = {spec.macroname: spec for spec in custom_arg_macros_text_specs_list}

            # Add specs for simple, common macros to ensure they output their LaTeX form
            # These will override preprocessor if preprocessor produces non-LaTeX or different LaTeX
            # For these, macroname is the command name without backslash.
            simple_latex_macros = {
                "Q": r"\mathbb{Q}", "C": r"\mathbb{C}", "K": r"\mathbb{K}",
                "F": r"\mathbb{F}", "N": r"\mathbb{N}", "Z": r"\mathbb{Z}",
                "R": r"\mathbb{R}", "A": r"\mathcal{A}", "V": r"\mathcal{V}",
                # Common Greek letters
                "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma", "delta": r"\delta",
                "epsilon": r"\epsilon", "zeta": r"\zeta", "eta": r"\eta", "theta": r"\theta",
                "iota": r"\iota", "kappa": r"\kappa", "lambda": r"\lambda", "mu": r"\mu",
                "nu": r"\nu", "xi": r"\xi", "pi": r"\pi", "rho": r"\rho",
                "sigma": r"\\sigma", "tau": r"\tau", "upsilon": r"\upsilon", "phi": r"\phi",
                "chi": r"\chi", "psi": r"\psi", "omega": r"\omega",
                "Gamma": r"\Gamma", "Delta": r"\Delta", "Theta": r"\Theta", "Lambda": r"\Lambda",
                "Xi": r"\Xi", "Pi": r"\Pi", "Sigma": r"\Sigma", "Upsilon": r"\Upsilon",
                "Phi": r"\Phi", "Psi": r"\Psi", "Omega": r"\Omega",
            }
            for name, repl_latex in simple_latex_macros.items():
                if name not in specs_map: # Add if not already defined (e.g. as an arg-taking macro)
                    specs_map[name] = MacroTextSpec(name, simplify_repl=repl_latex)
                else:
                    print(f"DEBUG CustomLatexNodes2Text: Macro '{name}' already had a spec, not overriding with simple LaTeX version.")

            final_custom_arg_specs = list(specs_map.values())

            if final_custom_arg_specs:
                category_name = 'custom_argument_macros_for_text_conversion_v6' # Incremented version
                self.latex_context_db.add_context_category(
                    category_name,
                    macros=final_custom_arg_specs,
                    prepend=True
                )
                print(f"DEBUG CustomLatexNodes2Text __init__: Category '{category_name}' (prepended) now has {len(final_custom_arg_specs)} MacroTextSpecs.")

            macros_to_remove_specs = [
                MacroTextSpec("title", simplify_repl=""), MacroTextSpec("author", simplify_repl=""),
                MacroTextSpec("date", simplify_repl=""), MacroTextSpec("maketitle", simplify_repl=""),
                MacroTextSpec("documentclass", simplify_repl=""), MacroTextSpec("usepackage", simplify_repl=""),
                MacroTextSpec("label", simplify_repl=""), MacroTextSpec("ref", simplify_repl=""),
                MacroTextSpec("cite", simplify_repl=""), MacroTextSpec("graphicspath", simplify_repl=""),
                MacroTextSpec("includegraphics", simplify_repl=""), MacroTextSpec("caption", simplify_repl=""),
                MacroTextSpec("newcommand", simplify_repl=""), MacroTextSpec("renewcommand", simplify_repl=""),
                MacroTextSpec("providecommand", simplify_repl=""), MacroTextSpec("DeclareMathOperator", simplify_repl=""),
                MacroTextSpec("newenvironment", simplify_repl=""), MacroTextSpec("renewenvironment", simplify_repl=""),
                MacroTextSpec("newtheorem", simplify_repl=""),
            ]
            if macros_to_remove_specs:
                remove_category_name = 'macros_to_explicitly_remove_v2'
                self.latex_context_db.add_context_category(
                    remove_category_name,
                    macros=macros_to_remove_specs,
                    prepend=True
                )
                print(f"DEBUG CustomLatexNodes2Text __init__: Category '{remove_category_name}' (prepended) now has {len(macros_to_remove_specs)} MacroTextSpecs for removal.")


        def convert_node(self, node):
            if node is None:
                return ""

            if node.isNodeType(LatexCharsNode):
                return node.chars
            
            if node.isNodeType(LatexMathNode):
                # node.delimiters stores the opening and closing delimiters, e.g., ('$', '$') or ('\\[', '\\]')
                # node.nodelist contains the parsed content *inside* the delimiters.
                # self.nodelist_to_text(node.nodelist) will convert this content.
                # Since math_mode='verbatim', this should give the LaTeX of the content.
                math_content_str = self.nodelist_to_text(node.nodelist)
                
                if node.delimiters == ('\\[', '\\]'):
                    return f"$${math_content_str}$$"
                elif node.delimiters == ('$', '$'):
                    return f"${math_content_str}$"
                else:
                    # Fallback for other delimiters or if delimiters are None (e.g. from macro expansion)
                    # If it's already valid LaTeX math, it should be fine.
                    # This path could be hit if a macro expands to math content without explicit delimiters.
                    # node.latex_verbatim() might be more robust here if available and gives full original string.
                    # However, math_content_str is from nodelist_to_text, which respects math_mode='verbatim'.
                    # If math_content_str is already $...$ or $$...$$, we don't want to double wrap.
                    # This case is tricky. For now, assume math_content_str is the core LaTeX.
                    # A check could be added: if not (math_content_str.startswith('$') and math_content_str.endswith('$'))
                    # This is imperfect. Let's trust math_content_str for now.
                    return math_content_str


            if node.isNodeType(LatexMacroNode):
                # print(f"DEBUG convert_node: Macro '{node.macroname}'")
                if node.macroname in ['textit', 'textbf', 'emph']:
                    if node.nodeargs and len(node.nodeargs) > 0 and \
                       node.nodeargs[0] is not None and hasattr(node.nodeargs[0], 'nodelist'):
                        return self.nodelist_to_text(node.nodeargs[0].nodelist)
                    return ""
                # The MacroTextSpecs for removal and custom handling should be triggered by super().convert_node()
                # The DEFINING_MACROS_TO_REMOVE set is a fallback.
                if node.macroname in DEFINING_MACROS_TO_REMOVE:
                    # Check if a specific MacroTextSpec handled it first (super() would not be called if this returns early)
                    # This path should ideally not be hit for macros covered by MacroTextSpec("", simplify_repl="")
                    print(f"DEBUG convert_node: Direct removal of macro '{node.macroname}' via DEFINING_MACROS_TO_REMOVE set (may indicate MacroTextSpec for removal was not hit or defined).")
                    return ""
                return super().convert_node(node)

            if node.isNodeType(LatexCommentNode):
                return ""
            
            if node.isNodeType(LatexEnvironmentNode):
                env_name = node.environmentname
                # Common LaTeX math environments
                math_env_names = { 
                    'equation', 'equation*', 'align', 'align*', 'alignat', 'alignat*',
                    'gather', 'gather*', 'multline', 'multline*', 'flalign', 'flalign*',
                    'displaymath' # although \[...\] is usually LatexMathNode
                }

                if env_name in math_env_names:
                    env_content_str = self.nodelist_to_text(node.nodelist)
                    # Reconstruct the full environment and wrap with $$ to ensure it's display math
                    # and to be consistent with how \[...\] is handled.
                    return f"$$\\begin{{{env_name}}}\n{env_content_str}\n\\end{{{env_name}}}$$"
                
                if env_name == 'document':
                    return self.nodelist_to_text(node.nodelist)
                
                if env_name in ['itemize', 'enumerate', 'description']:
                    return super().convert_node(node) # Let pylatexenc handle list formatting

                environments_to_process_content = [
                    'abstract', 'center', 'figure', 'table', # Note: figure/table content extraction is basic
                    'theorem', 'lemma', 'proof', 'definition', 'corollary', 'example', 'remark'
                ]
                if env_name in environments_to_process_content:
                    return self.nodelist_to_text(node.nodelist)

                environments_to_remove = ['comment'] # LaTeX 'comment' environment
                if env_name in environments_to_remove:
                    return ""
                
                # Fallback for other environments
                return super().convert_node(node)

            if node.isNodeType(LatexGroupNode):
                return self.nodelist_to_text(node.nodelist)

            return super().convert_node(node)

    if not nodelist:
        print("DEBUG custom_latex_to_text: Nodelist is empty after parsing attempt.")
        return ""

    converter = CustomLatexNodes2Text()
    text_content = converter.nodelist_to_text(nodelist)

    if text_content:
        text_content = text_content.replace('\r\n', '\n').replace('\r', '\n')
        lines = text_content.split('\n')
        processed_lines = []
        for line in lines:
            stripped_line = line.strip()
            # Basic handling for lines that look like list items from default pylatexenc output
            list_marker_match = re.match(r"^(\s*)(?:[\*\-]\s+|\d+\.\s+)(.*)", stripped_line)
            if list_marker_match:
                marker_part = stripped_line.split(None, 1)[0]
                item_content = list_marker_match.group(2).strip()
                if item_content:
                    processed_lines.append(f"{marker_part} {item_content}")
            elif stripped_line:
                processed_lines.append(stripped_line)
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
        latex_content_preprocessed = _preprocess_custom_latex_commands(latex_content_original)
        if latex_content_preprocessed != latex_content_original:
            print("DEBUG parse_latex_file: Content was modified by preprocessor.")
        else:
            print("DEBUG parse_latex_file: Content was NOT modified by preprocessor.")

        text_representation = custom_latex_to_text(latex_content_preprocessed)

        if not text_representation.strip() and latex_content_original.strip():
            print(f"INFO: custom_latex_to_text returned empty/whitespace for non-empty file: {file_path}. This might be due to parsing errors or aggressive filtering.")

        return text_representation

    except RecursionError as re_err:
        print(f"ERROR: Recursion depth exceeded at parse_latex_file level for {file_path}. This is unexpected here. Error: {re_err}")
        return ""
    except FileNotFoundError:
        print(f"ERROR: LaTeX file not found: {file_path}")
        return ""
    except Exception as e:
        print(f"ERROR: Unexpected error parsing LaTeX file {file_path}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return ""

if __name__ == '__main__':
    class DummyConfig:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        RAW_LATEX_DIR = os.path.join(BASE_DIR, 'data', 'raw_latex')
        PARSED_LATEX_OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'parsed_content', 'from_latex')

    cfg_to_use = DummyConfig()

    if config: # Use actual config if available
        raw_latex_dir_to_use = getattr(config, 'DATA_DIR_RAW_LATEX', cfg_to_use.RAW_LATEX_DIR)
        if hasattr(config, 'DATA_DIR_PARSED_LATEX'):
            parsed_latex_output_dir_to_use = config.DATA_DIR_PARSED_LATEX
        elif hasattr(config, 'PARSED_CONTENT_DIR'):
            parsed_latex_output_dir_to_use = os.path.join(config.PARSED_CONTENT_DIR, 'from_latex')
        else:
            parsed_latex_output_dir_to_use = cfg_to_use.PARSED_LATEX_OUTPUT_DIR
    else: # Fallback to dummy config
        raw_latex_dir_to_use = cfg_to_use.RAW_LATEX_DIR
        parsed_latex_output_dir_to_use = cfg_to_use.PARSED_LATEX_OUTPUT_DIR

    print(f"--- Processing all .tex files in: {raw_latex_dir_to_use} ---")
    print(f"--- Parsed output will be saved to: {parsed_latex_output_dir_to_use} ---")

    if not os.path.exists(raw_latex_dir_to_use):
        print(f"ERROR: Raw LaTeX directory not found: {raw_latex_dir_to_use}")
        sys.exit(1)
    if not os.path.exists(parsed_latex_output_dir_to_use):
        try:
            os.makedirs(parsed_latex_output_dir_to_use)
            print(f"Created output directory: {parsed_latex_output_dir_to_use}")
        except OSError as e:
            print(f"ERROR: Could not create output directory {parsed_latex_output_dir_to_use}: {e}")
            sys.exit(1)

    try:
        tex_files_in_dir = [f for f in os.listdir(raw_latex_dir_to_use) if f.endswith(".tex")]
    except FileNotFoundError:
        print(f"ERROR: Raw LaTeX directory not found when listing files: {raw_latex_dir_to_use}")
        sys.exit(1)

    if not tex_files_in_dir:
        print(f"No .tex files found in {raw_latex_dir_to_use}.")
        dummy_file_path = os.path.join(raw_latex_dir_to_use, "dummy_example.tex")
        try:
            with open(dummy_file_path, "w", encoding="utf-8") as df:
                df.write("\\documentclass{article}\n")
                df.write("\\title{Dummy Title Test}\n\\author{Dummy Author}\n\\date{Today}\n")
                df.write("\\usepackage{amsmath}\n")
                df.write("\\newcommand{\\mycmd}{My Custom Command Text}\n")
                df.write("\\newcommand{\\Q}{\\mathbb{Q}}\n")
                df.write("\\newcommand{\\abs}[1]{\\left| #1 \\right|}\n")
                df.write("\\newcommand{\\unbalancedcmd}{{unbalanced_content}}\n")
                df.write("\\DeclareMathOperator{\\TestOp}{TestOp}\n")
                df.write("\\begin{document}\n")
                df.write("\\maketitle\n")
                df.write("Hello, world! $\\Q$. Test $\\abs{-5}$. \\mycmd. \\unbalancedcmd. $\\TestOp(x) = y$.\n")
                df.write("Display math: \\<y_bin_473>\\alpha + \\beta = \\gamma\\]\n")
                df.write("\\begin{align*} x &= y + z \\\\ a &= b \\end{align*}\n")
                df.write("\\section{Test Section}\nText in section.\n")
                df.write("\\begin{itemize}\\item First item.\\item Second item.\\end{itemize}\n")
                df.write("\\end{document}\n")
            print(f"Created a dummy file for testing: {dummy_file_path}")
            tex_files_in_dir = ["dummy_example.tex"]
        except IOError as e:
            print(f"Could not create dummy .tex file: {e}")

    processed_count = 0
    successful_parses = 0
    for filename in tex_files_in_dir:
        if filename.startswith("._"):
            print(f"Skipping AppleDouble file: {filename}")
            continue

        file_path = os.path.join(raw_latex_dir_to_use, filename)
        print(f"\n--- Parsing: {filename} ---")
        parsed_content = parse_latex_file(file_path)

        if parsed_content and parsed_content.strip():
            print(f"--- Parsed Content from {filename} (first 500 chars) ---")
            print(parsed_content[:500] + ("..." if len(parsed_content) > 500 else ""))
            output_filename = os.path.splitext(filename)[0] + "_parsed.txt"
            output_path = os.path.join(parsed_latex_output_dir_to_use, output_filename)
            try:
                with open(output_path, "w", encoding="utf-8") as f_out:
                    f_out.write(parsed_content)
                print(f"Saved parsed LaTeX content to: {output_path}")
                successful_parses += 1
            except IOError as e:
                print(f"ERROR: Could not write parsed content to {output_path}: {e}")
        else:
            print(f"Could not parse or empty content for: {filename}")
        processed_count +=1

    print(f"\n--- Finished processing {processed_count} files. ---")
    if tex_files_in_dir:
        print(f"Successfully parsed and saved {successful_parses} out of {len(tex_files_in_dir)} .tex files found.")
        if successful_parses < len(tex_files_in_dir):
             print("WARNING: Some .tex files could not be parsed or resulted in empty content.")
    elif processed_count > 0 and successful_parses == 0 :
        print("WARNING: No .tex files were successfully parsed (though some files were attempted).")
    elif not tex_files_in_dir and processed_count == 0 :
        print(f"No .tex files found in {raw_latex_dir_to_use} and no dummy file was processed.")

