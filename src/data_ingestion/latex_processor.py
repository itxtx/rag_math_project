# src/data_ingestion/latex_processor.py
import re
from typing import Dict, Tuple, List, Optional
import argparse
import sys
import os

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

def replace_simple_command(text: str, cmd_name: str, replacement: str) -> str:
    """
    Replace a simple command without arguments.
    Handles various contexts where the command might appear.
    """
    # Main pattern: match \cmd not followed by alphanumeric characters
    pattern = f'\\\\{re.escape(cmd_name)}(?![a-zA-Z0-9])'
    
    # Escape backslashes in replacement for re.sub
    # This is crucial for replacements containing LaTeX commands
    escaped_replacement = replacement.replace('\\', r'\\')
    
    # Replace all occurrences
    result = re.sub(pattern, escaped_replacement, text)
    
    return result

def replace_command_with_args(text: str, cmd_name: str, replacement: str, num_args: int) -> str:
    """
    Replace a command that takes arguments.
    Example: \Zn{5} -> \Z_{5} where \Zn is defined as \Z_{#1}
    """
    if num_args == 0:
        # Simple replacement without arguments
        return replace_simple_command(text, cmd_name, replacement)
    
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
        # For more than 2 arguments, build a dynamic pattern
        arg_pattern = '\\{([^}]*)\\}'
        full_pattern = f'\\\\{re.escape(cmd_name)}' + (arg_pattern * num_args)
        
        def replace_func(match):
            result = replacement
            for i in range(1, num_args + 1):
                result = result.replace(f'#{i}', match.group(i))
            return result
        
        return re.sub(full_pattern, replace_func, text)

def process_latex_document(input_text: str, max_iterations: int = 10, debug: bool = False) -> str:
    """
    Process a LaTeX document by replacing all defined commands with their expansions.
    Performs multiple passes to handle nested command replacements.
    """
    commands = extract_commands(input_text)
    
    if not commands:
        print("No command definitions found by latex_processor.")
        return input_text

    print(f"latex_processor: Found {len(commands)} command definitions:")
    for cmd_name, cmd_info in commands.items():
        if cmd_info['type'] == 'newcommand':
            args_info = f" (takes {cmd_info['num_args']} args)" if cmd_info['num_args'] > 0 else ""
            print(f"  \\{cmd_name} -> {cmd_info['replacement']}{args_info}")
        else:
            print(f"  \\{cmd_name} -> \\operatorname{{{cmd_info['replacement']}}}")
    
    # Sort commands by key length (descending) to replace longer command names first
    sorted_cmd_names = sorted(commands.keys(), key=len, reverse=True)
    
    result_text = input_text
    
    # Perform multiple passes to handle nested replacements
    for iteration in range(max_iterations):
        text_before = result_text
        replacements_made = False
        
        for cmd_name in sorted_cmd_names:
            cmd_info = commands[cmd_name]
            old_text = result_text
            
            if cmd_info['type'] == 'newcommand':
                result_text = replace_command_with_args(
                    result_text, 
                    cmd_name, 
                    cmd_info['replacement'], 
                    cmd_info['num_args']
                )
            elif cmd_info['type'] == 'mathoperator':
                # For \DeclareMathOperator{\foo}{bar}, replace \foo with \operatorname{bar}
                operatorname_replacement = f"\\operatorname{{{cmd_info['replacement']}}}"
                result_text = replace_simple_command(result_text, cmd_name, operatorname_replacement)
            
            if old_text != result_text:
                replacements_made = True
                if debug:
                    print(f"  Iteration {iteration + 1}: Applied \\{cmd_name}")
        
        # If no replacements were made in this iteration, we're done
        if not replacements_made or text_before == result_text:
            if debug:
                print(f"  Completed after {iteration + 1} iteration(s)")
            break
        
        if iteration == max_iterations - 1:
            print(f"Warning: Reached maximum iterations ({max_iterations}). Some nested commands may not be fully expanded.")
    
    return result_text

def remove_command_definitions(text: str) -> str:
    """
    Remove the \newcommand and \DeclareMathOperator definition lines from the text.
    """
    lines = text.split('\n')
    filtered_lines = []
    
    # Pattern to match definition lines
    command_def_pattern = re.compile(r'^\s*\\(newcommand|renewcommand|DeclareMathOperator)\s*\{')
    
    for line in lines:
        if not command_def_pattern.match(line.strip()):
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def main_cli():
    parser = argparse.ArgumentParser(description='Replace LaTeX custom commands with their definitions')
    parser.add_argument('input_file', help='Input LaTeX file')
    parser.add_argument('-o', '--output', help='Output file (default: input_file_expanded.tex)')
    parser.add_argument('--keep-definitions', action='store_true',
                        help='Keep the original command definitions in the output')
    parser.add_argument('--max-iterations', type=int, default=10,
                        help='Maximum iterations for nested replacements (default: 10)')
    
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

    processed_text = process_latex_document(input_text, args.max_iterations)
    
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
    if len(sys.argv) > 1:  # If command line arguments are provided
        sys.exit(main_cli())
    else:  # Demo mode
        print("Demo mode - processing sample LaTeX snippet with complex cases...\n")
        
        sample_text = r"""% Preamble
\documentclass{article}
\usepackage{amsmath}
% Basic field symbols
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\F}{\mathbb{F}}

% Math operators
\DeclareMathOperator{\Aff}{Aff}
\DeclareMathOperator{\lcm}{lcm}
\DeclareMathOperator{\Ker}{Ker}
\DeclareMathOperator{\im}{im}
\DeclareMathOperator{\kerop}{ker}
\DeclareMathOperator{\dimv}{dim}

% Field theory commands
\newcommand{\fieldext}[2]{{#1/#2}} % Field extension K/F
\newcommand{\degree}[2]{[#1:#2]} % Degree of extension [K:F]
\newcommand{\Zn}[1]{\Z_{#1}}
\newcommand{\Znx}[1]{\Z_{#1}^\times}
\newcommand{\degpoly}{\operatorname{deg}}
\newcommand{\ideal}[1]{\langle #1 \rangle}
\newcommand{\abs}[1]{|#1|}
\newcommand{\norm}[1]{\text{N}(#1)}
\newcommand{\kerphi}{\operatorname{ker}\phi}
\newcommand{\imphi}{\operatorname{im}\phi}
\newcommand{\Pn}{P_n(\F)}

% Functional analysis commands
\newcommand{\mathcalB}{\mathcal{L}} % Changed to L for consistency with L(E,F)
\newcommand{\id}{\mathrm{Id}} % Identity operator
\newcommand{\e}{\mathrm{e}} % Euler's number
\newcommand{\Lop}{\mathcal{L}} % Space of bounded linear operators
\newcommand{\dist}{d} % Use \dist for the metric
\newcommand{\Tau}{\mathcal{T}} % Topology
\newcommand{\dual}[1]{#1^*} % Dual space
\newcommand{\Distr}{\mathcal{D}'} % Distributions
\newcommand{\TestFunc}{\mathcal{D}} % Test functions
\newcommand{\Lp}[1]{L^{#1}} % L^p space
\newcommand{\lp}[1]{\ell^{#1}} % l^p space
\newcommand{\linop}[2]{\mathcal{L}(#1, #2)} % Space of linear operators
\newcommand{\boundedlinop}[1]{\mathcal{L}(#1)} % Space of bounded linear operators on E

% Calculus commands
\newcommand{\diff}[2]{\frac{d#1}{d#2}}
\newcommand{\pdiff}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\udot}{\dot{u}} % Example using dot notation for time derivative
\newcommand{\xdot}{\dot{x}} % Example using dot notation for time derivative
\newcommand{\bvec}[1]{\bm{#1}} % For bold vectors

% Document Body
\begin{document}
Basic test cases:
- \operatorname{Aff}(\R) should become \operatorname{Aff}(\mathbb{R})
- \R} should become \mathbb{R}}
- \operatorname{lcm}(|6|_{\Z_8}) should work with nested \Z
- $\operatorname{Ker}(\phi) = \{x \in \Z \mid 3x \equiv 0 \pmod{12}\}$
- $\phi: (\C^\times, \times) \to (\R_{>0}, \times)$

Field extension test cases:
- The extension $\fieldext{K}{F}$ has degree $\degree{K}{F}$
- Example: $\fieldext{\C}{\R}$ with $\degree{\C}{\R} = \dimv_\R \C = 2$
- Another: $\fieldext{\Q(\sqrt{2})}{\Q}$ with degree $\degree{\Q(\sqrt{2})}{\Q} = 2$
- Complex nested: $\fieldext{\Q(\sqrt{2}, \sqrt{3})}{\Q}$

Polynomial and algebra cases:
- If $\degpoly(r(x)) < \degpoly(d(x)) = n$
- The polynomial space $\Pn$ contains polynomials of degree at most $n$
- The ideal $\ideal{x, y, z}$ in the ring $\R[x,y,z]$

Functional analysis cases:
- The operator space $\linop{E}{F}$ or simply $\mathcalB(E,F)$
- The bounded operators $\boundedlinop{H}$ on Hilbert space $H$
- The identity operator $\id$ on $\Lop(E)$
- The dual space $\dual{E}$ of $E$
- Distribution space $\Distr$ and test functions $\TestFunc$
- $\Lp{p}$ spaces for $1 \leq p \leq \infty$
- Sequence spaces $\lp{2}$ and $\lp{\infty}$
- Metric: $\dist(x,y) < \epsilon$ in topology $\Tau$

Calculus cases:
- Derivatives: $\diff{y}{x} = f'(x)$ and $\pdiff{u}{x} + \pdiff{u}{y} = 0$
- Time derivatives: $\xdot = v$ and $\udot = -ku$
- Vector notation: $\bvec{v} = \bvec{e}_1 + 2\bvec{e}_2$
- Euler's number: $\e^{i\pi} + 1 = 0$

Mixed complex cases:
- The kernel $\kerop(A) = \{ \mathbf{x} \in \F^n \mid A\mathbf{x} = \mathbf{0} \}$
- The column space $\im(A) = \{ A\mathbf{x} \mid \mathbf{x} \in \F^n \}$
- Norm and absolute value: $\norm{\bvec{x}} \leq \abs{\lambda} \norm{\bvec{y}}$
- Field extension degree: $\degree{L}{F} = \degree{L}{K} \cdot \degree{K}{F}$
\end{document}"""
        
        print("=" * 70)
        print("Original text:")
        print("=" * 70)
        print(sample_text)
        
        print("\n" + "=" * 70)
        print("Processing with multiple iterations to handle nested replacements...")
        print("=" * 70)
        
        expanded_text = process_latex_document(sample_text, debug=False)
        
        print("\n" + "=" * 70)
        print("Text after command expansion:")
        print("=" * 70)
        print(expanded_text)
        
        final_text = remove_command_definitions(expanded_text)
        print("\n" + "=" * 70)
        print("Final text after removing definitions:")
        print("=" * 70)
        print(final_text)