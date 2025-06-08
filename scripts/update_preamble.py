# scripts/update_preamble.py
import os
import sys
import argparse
import re
from datetime import datetime
from typing import Optional, Tuple, Dict
# --- Setup Project Path ---
# This allows the script to import modules from the 'src' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)



# --- Configuration ---
DEFAULT_PREAMBLE_PATH = os.path.join(project_root, 'data', 'master_preamble.tex')
def parse_newcommand(line: str) -> Tuple[Optional[str], Optional[str], int]:
    """
    Parse a \newcommand definition.
    Returns (command_name, replacement, num_args) or (None, None, 0) if no match.
    """
    # Match \newcommand{\cmdname}[num_args]{replacement} or \newcommand{\cmdname}{replacement}
    # This pattern handles multi-line and complex replacements better
    pattern = r'\\newcommand\s*\{\s*\\([^}]+)\s*\}\s*(?:\[\s*(\d+)\s*\])?\s*\{(.*)\}\s*$'
    
    match = re.match(pattern, line.strip())
    if match:
        cmd_name = match.group(1)
        num_args_str = match.group(2)
        num_args = int(num_args_str) if num_args_str else 0
        replacement = match.group(3)
        return cmd_name, replacement, num_args
    return None, None, 0

def parse_declaremathoperator(line: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a \DeclareMathOperator definition.
    Returns (command_name, operator_name) or (None, None) if no match.
    """
    # Match \DeclareMathOperator{\cmdname}{opname}
    pattern = r'\\DeclareMathOperator\s*\{\s*\\([^}]+)\s*\}\s*\{([^}]*)\}\s*$'
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
                    'replacement': op_name,
                    'defined_at_line': line_number + 1
                }
    
    return commands
def get_existing_command_names(preamble_path: str) -> set:
    """Reads the preamble file and returns a set of already defined command names."""
    if not os.path.exists(preamble_path):
        return set()
    
    with open(preamble_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # This regex finds the command name (e.g., 'R', 'fieldext') from definitions.
    pattern = re.compile(r'\\(?:newcommand|DeclareMathOperator)\s*\{\s*\\([^}]+)')
    found_commands = pattern.findall(content)
    
    return set(found_commands)

def find_new_command_definitions(source_file_path: str, existing_commands: set) -> list:
    """
    Extracts command definitions from a source file and returns the full line
    of any command that is not in the existing_commands set.
    """
    new_definition_lines = []
    
    with open(source_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        source_content = f.read()

    # Reuse the robust extraction logic from your data ingestion pipeline
    all_commands_in_source = extract_commands(source_content)
    
    # We need the original lines to append them verbatim
    source_lines = source_content.split('\n')
    
    for cmd_name, cmd_info in all_commands_in_source.items():
        if cmd_name not in existing_commands:
            # The 'defined_at_line' is 1-indexed, so subtract 1 for list index
            line_index = cmd_info.get('defined_at_line', 0) - 1
            
            if 0 <= line_index < len(source_lines):
                definition_line = source_lines[line_index].strip()
                # Simple check to make sure we have the correct definition line
                if definition_line.startswith(r'\%') or not definition_line:
                    continue # Skip commented out or empty lines
                
                new_definition_lines.append(definition_line)
                print(f"  Found new command: \\{cmd_name}")

    return new_definition_lines

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Finds new custom LaTeX commands in a source file and adds them to the master preamble.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "source_file",
        help="Path to the .tex file to scan for new commands."
    )
    parser.add_argument(
        "--preamble",
        default=DEFAULT_PREAMBLE_PATH,
        help=f"Path to the master preamble file.\n(default: {DEFAULT_PREAMBLE_PATH})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print new commands that would be added, but do not modify the preamble file."
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source_file):
        print(f"Error: Source file not found at '{args.source_file}'")
        sys.exit(1)
        
    print(f"Reading existing commands from: {args.preamble}")
    existing_commands = get_existing_command_names(args.preamble)
    print(f"Found {len(existing_commands)} existing command definitions.")
    
    print(f"\nScanning for new command definitions in: {args.source_file}")
    new_definitions_to_add = find_new_command_definitions(args.source_file, existing_commands)
    
    if not new_definitions_to_add:
        print("\n✅ No new commands found. Preamble is up to date.")
        sys.exit(0)
    
    print(f"\nDiscovered {len(new_definitions_to_add)} new command definition(s) to add:")
    for line in new_definitions_to_add:
        print(f"  + {line}")
        
    if args.dry_run:
        print("\n-- DRY RUN -- Preamble file was not modified.")
        sys.exit(0)
        
    print(f"\nAppending new commands to {args.preamble}...")
    try:
        with open(args.preamble, 'a', encoding='utf-8') as f:
            f.write("\n\n% --- Commands auto-added from {} on {} ---\n".format(
                os.path.basename(args.source_file),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            for line in new_definitions_to_add:
                f.write(line + "\n")
        print("✅ Successfully updated preamble.")
    except Exception as e:
        print(f"Error: Failed to write to preamble file. {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
