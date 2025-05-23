# src/data_ingestion/latex_processor.py
import re
from typing import Dict, Tuple, List
import argparse
import sys # Added for sys.exit in main
import os
from typing import Optional

def parse_newcommand(line: str) -> Optional[Tuple[str, str, int]]:
    """
    Parse a \newcommand definition.
    Returns (command_name, replacement, num_args) or (None, None, 0) if no match.
    """
    # Match \newcommand{\cmdname}[num_args]{replacement} or \newcommand{\cmdname}{replacement}
    pattern = r'\\newcommand\s*\{\s*\\([^}]+)\s*\}\s*(?:\[\s*(\d+)\s*\])?\s*\{(.*)\}'
    # Handle potential escaped braces in the replacement part correctly by making the last group non-greedy
    # and ensuring we match the final closing brace correctly. This can be tricky if '}' appears in replacement.
    # A more robust parser might be needed for very complex replacements.
    # For now, assume replacement doesn't contain unmatched '}'.
    pattern_safer = r'\\newcommand\s*\{\s*\\([^}]+)\s*\}\s*(?:\[\s*(\d+)\s*\])?\s*\{(.*)\}\s*$'
    # Using original pattern as it's simpler and often works for common cases.
    # Complex replacements inside {} might require more advanced parsing.

    match = re.match(pattern, line.strip())
    if match:
        cmd_name = match.group(1)
        num_args_str = match.group(2)
        num_args = int(num_args_str) if num_args_str else 0
        replacement = match.group(3) # This might contain #1, #2 etc.
        return cmd_name, replacement, num_args
    return None, None, 0

def parse_declaremathoperator(line: str) -> Optional[Tuple[str, str]]:
    """
    Parse a \DeclareMathOperator definition.
    Returns (command_name, operator_name) or (None, None) if no match.
    """
    # Match \DeclareMathOperator{\cmdname}{opname}
    pattern = r'\\DeclareMathOperator\s*\{\s*\\([^}]+)\s*\}\s*\{(.*?)\}\s*$' # Added \s*$ for robustness
    match = re.match(pattern, line.strip())
    if match:
        cmd_name = match.group(1)
        op_name = match.group(2)
        return cmd_name, op_name
    return None, None

def extract_commands(text: str) -> Dict[str, Dict]:
    """
    Extract all \newcommand and \DeclareMathOperator definitions from text.
    Returns a dictionary of command definitions.
    """
    commands = {}
    lines = text.split('\n')
    for line_number, line_content in enumerate(lines):
        line = line_content.strip()

        # Skip comments and empty lines
        if not line or line.startswith('%'):
            continue

        # Parse \newcommand
        if line.startswith('\\newcommand'):
            cmd_name, replacement, num_args = parse_newcommand(line)
            if cmd_name:
                if cmd_name in commands:
                    print(f"Warning: Command \\{cmd_name} redefined on line {line_number + 1}. Using new definition.")
                commands[cmd_name] = {
                    'type': 'newcommand',
                    'replacement': replacement,
                    'num_args': num_args,
                    'defined_at_line': line_number + 1
                }
        # Parse \DeclareMathOperator
        elif line.startswith('\\DeclareMathOperator'):
            cmd_name, op_name = parse_declaremathoperator(line)
            if cmd_name:
                if cmd_name in commands:
                    print(f"Warning: Command \\{cmd_name} (as DeclareMathOperator) redefined on line {line_number + 1}. Using new definition.")
                commands[cmd_name] = {
                    'type': 'mathoperator',
                    'replacement': op_name, # Store the operator name itself, not f'\\operatorname{{{op_name}}}' yet
                    'defined_at_line': line_number + 1
                }
    return commands

def replace_command_with_args(text: str, cmd_name: str, replacement: str, num_args: int) -> str:
    """
    Replace a command that takes arguments.
    Example: \Zn{5} -> \Z_{5} where \Zn is defined as \Z_{#1}
    """
    if num_args == 0:
        # Simple replacement without arguments
        pattern = f'\\\\{re.escape(cmd_name)}(?![a-zA-Z])'
        return re.sub(pattern, replacement, text)
    
    elif num_args == 1:
        # Command with one argument: \cmd{arg}
        pattern = f'\\\\{re.escape(cmd_name)}\\{{([^}}]*)\\}}'
        
        def replace_func(match):
            arg = match.group(1)
            return replacement.replace('#1', arg)
        
        return re.sub(pattern, replace_func, text)
    
    elif num_args == 2:
        # Command with two arguments: \cmd{arg1}{arg2}
        pattern = f'\\\\{re.escape(cmd_name)}\\{{([^}}]*)\\}}\\{{([^}}]*)\\}}'
        
        def replace_func(match):
            arg1, arg2 = match.group(1), match.group(2)
            result = replacement.replace('#1', arg1).replace('#2', arg2)
            return result
        
        return re.sub(pattern, replace_func, text)
    
    else:
        # For more than 2 arguments, we'd need a more complex pattern
        print(f"Warning: Command \\{cmd_name} has {num_args} arguments, which is not fully supported")
        return text


def replace_simple_command(text: str, cmd_name: str, replacement: str) -> str:
    """
    Replace a simple command without arguments (or a DeclareMathOperator).
    """
    # Replacement must be escaped for re.sub if it contains backslashes
    escaped_replacement = replacement.replace('\\', '\\\\')
    pattern = f'\\\\{re.escape(cmd_name)}(?![a-zA-Z])' # Match \cmdname not followed by a letter
    return re.sub(pattern, escaped_replacement, text)

def process_latex_document(input_text: str) -> str:
    """
    Process a LaTeX document by replacing all defined commands with their expansions.
    """
    commands = extract_commands(input_text)
    
    if not commands:
        print("No command definitions found by latex_processor.")
        return input_text

    print(f"latex_processor: Found {len(commands)} command definitions:")
    # Sort commands by key length (descending) to replace longer command names first
    # This helps with commands that might be prefixes of others (e.g., \cmd vs \cmdlong)
    # Although the (?![a-zA-Z]) lookahead helps, sorting is an added precaution.
    sorted_cmd_names = sorted(commands.keys(), key=len, reverse=True)

    result_text = input_text
    
    for cmd_name in sorted_cmd_names:
        cmd_info = commands[cmd_name]
        print(f"  Processing definition for: \\{cmd_name}")
        
        if cmd_info['type'] == 'newcommand':
            if cmd_info['num_args'] == 0:
                # Simple replacement, definition is already the text to insert
                result_text = replace_simple_command(result_text, cmd_name, cmd_info['replacement'])
            else:
                # Argument-taking command
                result_text = replace_command_with_args(
                    result_text, 
                    cmd_name, 
                    cmd_info['replacement'], 
                    cmd_info['num_args']
                )
        elif cmd_info['type'] == 'mathoperator':
            # For \DeclareMathOperator{\foo}{bar}, replace \foo with \operatorname{bar}
            # The replacement from extract_commands for mathoperator is just "bar"
            # We need to wrap it with \operatorname{}
            operatorname_replacement = f"\\operatorname{{{cmd_info['replacement']}}}"
            result_text = replace_simple_command(result_text, cmd_name, operatorname_replacement)
            
    return result_text

def remove_command_definitions(text: str) -> str:
    """
    Remove the \newcommand and \DeclareMathOperator definition lines from the text.
    """
    lines = text.split('\n')
    filtered_lines = []
    # More specific regex to match definition lines correctly
    command_def_pattern = re.compile(r"^\s*\\(newcommand|renewcommand|DeclareMathOperator)\s*\{.*")

    for line in lines:
        if not command_def_pattern.match(line.strip()):
            filtered_lines.append(line)
    return '\n'.join(filtered_lines)

def main_cli(): # Renamed to avoid conflict with any 'main' in calling scripts
    parser = argparse.ArgumentParser(description='Replace LaTeX custom commands with their definitions')
    parser.add_argument('input_file', help='Input LaTeX file')
    parser.add_argument('-o', '--output', help='Output file (default: input_file_expanded.tex)')
    parser.add_argument('--keep-definitions', action='store_true',
                        help='Keep the original command definitions in the output')
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_text = f.read()
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found")
        return 1
    except Exception as e:
        print(f"Error reading file: {e}")
        return 1

    processed_text = process_latex_document(input_text)
    if not args.keep_definitions:
        processed_text = remove_command_definitions(processed_text)

    output_file = args.output if args.output else f"{os.path.splitext(args.input_file)[0]}_expanded.tex"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(processed_text)
        print(f"\nProcessed document saved to: {output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")
        return 1
    return 0

if __name__ == "__main__":
    # Example usage if run directly
    if len(sys.argv) > 1 : # If command line arguments are provided
        sys.exit(main_cli())
    else: # Demo mode
        print("Demo mode - processing sample LaTeX snippet...")
        sample_text = """
% Preamble
\\documentclass{article}
\\usepackage{amsmath}
\\newcommand{\\R}{\\mathbb{R}}
\\newcommand{\Zn}[1]{\\Z_{#1}}
\\DeclareMathOperator{\\Ker}{Ker}
\\newcommand{\\vec}[1]{\\mathbf{#1}}
\\newcommand{\\simple}{SimpleText}

% Document Body
\\begin{document}
This is some text with math: $\\R$.
We consider the group $\\Zn{5}$.
The kernel is denoted $\\Ker A$.
A vector is $\\vec{v}$.
This is just \\simple.
\\end{document}
        """
        print("\nOriginal text:")
        print(sample_text)
        
        expanded_text = process_latex_document(sample_text)
        print("\nText after command expansion:")
        print(expanded_text)
        
        final_text = remove_command_definitions(expanded_text)
        print("\nFinal text after removing definitions:")
        print(final_text)
