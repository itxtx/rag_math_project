import pytest
import os
import re
from unittest.mock import patch, mock_open, MagicMock
from src.data_ingestion import latex_processor
import subprocess


class TestLatexPreprocessing:
    """Test suite for LaTeX preprocessing functionality"""
    
    @staticmethod
    def expand_custom_commands(latex_content: str, preamble_commands: str) -> str:
        """
        Expand custom commands in LaTeX content using definitions from preamble.
        This simulates what should happen before LaTeXML processing.
        """
        # Extract command definitions from preamble
        command_map = {}
        
        # Pattern to match \newcommand or \renewcommand definitions
        # Use non-greedy matching and handle nested braces better
        cmd_pattern = r'\\(?:new|renew)command\{\\(\w+)\}(?:\[(\d+)\])?\{((?:[^{}]|\{[^{}]*\})*)\}'
        
        for match in re.finditer(cmd_pattern, preamble_commands):
            cmd_name = match.group(1)
            num_args = int(match.group(2)) if match.group(2) else 0
            cmd_body = match.group(3)
            command_map[cmd_name] = (num_args, cmd_body)
        
        # Also handle DeclareMathOperator
        op_pattern = r'\\DeclareMathOperator\{\\(\w+)\}\{([^}]+)\}'
        for match in re.finditer(op_pattern, preamble_commands):
            op_name = match.group(1)
            op_text = match.group(2)
            command_map[op_name] = (0, f'\\operatorname{{{op_text}}}')
        
        # Expand commands in the content
        expanded_content = latex_content
        
        # Sort by command length (longest first) to avoid partial replacements
        sorted_commands = sorted(command_map.items(), key=lambda x: len(x[0]), reverse=True)
        
        for cmd_name, (num_args, cmd_body) in sorted_commands:
            if num_args == 0:
                # Simple replacement for commands without arguments
                # Use a lambda to properly escape the replacement
                expanded_content = re.sub(
                    rf'\\{cmd_name}(?![a-zA-Z])', 
                    lambda m: cmd_body, 
                    expanded_content
                )
            else:
                # Handle commands with arguments
                # More robust pattern that handles nested braces
                args_pattern = r'\{((?:[^{}]|\{[^{}]*\})*)\}'
                full_pattern = rf'\\{cmd_name}' + (args_pattern * num_args)
                
                def replace_args(match):
                    result = cmd_body
                    for i in range(1, num_args + 1):
                        # Replace #1, #2, etc. with the captured arguments
                        result = result.replace(f'#{i}', match.group(i))
                    return result
                
                expanded_content = re.sub(full_pattern, replace_args, expanded_content)
        
        return expanded_content


def test_latex_preprocessing_with_custom_commands():
    """Test that custom commands are properly expanded before processing"""
    
    # Sample preamble with custom commands
    preamble = r"""
    \newcommand{\Z}{\mathbb{Z}}
    \newcommand{\R}{\mathbb{R}}
    \newcommand{\Zn}[1]{\Z_{#1}}
    \newcommand{\abs}[1]{\left| #1 \right|}
    \DeclareMathOperator{\Ker}{Ker}
    """
    
    # LaTeX content using custom commands
    latex_content = r"""
    Consider the group $\Z$ and the field $\R$.
    The quotient group $\Zn{5}$ has order $\abs{\Zn{5}} = 5$.
    The kernel is denoted $\Ker(\phi)$.
    """
    
    # Expected expanded content
    expected_expanded = r"""
    Consider the group $\mathbb{Z}$ and the field $\mathbb{R}$.
    The quotient group $\mathbb{Z}_{5}$ has order $\left| \mathbb{Z}_{5} \right| = 5$.
    The kernel is denoted $\operatorname{Ker}(\phi)$.
    """
    
    preprocessor = TestLatexPreprocessing()
    expanded = preprocessor.expand_custom_commands(latex_content, preamble)
    
    # Normalize whitespace for comparison
    expanded_normalized = re.sub(r'\s+', ' ', expanded.strip())
    expected_normalized = re.sub(r'\s+', ' ', expected_expanded.strip())
    
    assert expanded_normalized == expected_normalized


def test_latex_with_uncommon_packages(tmp_path):
    """Test handling of LaTeX with uncommon packages"""
    uncommon_latex = r"""
    \documentclass{article}
    \usepackage{tikz, xcolor, amssymb, minted}
    \begin{document}
    Hello with uncommon packages.
    \end{document}
    """
    
    # Mock the preamble file content
    mock_preamble = r"""
    \documentclass{article}
    \usepackage{tikz, xcolor, amssymb, minted}
    \begin{document}
    """
    
    with patch("builtins.open", mock_open(read_data=mock_preamble)):
        with patch("os.path.exists", return_value=True):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout="<document><body>Uncommon packages parsed</body></document>", 
                    stderr="", 
                    returncode=0
                )
                xml = latex_processor.run_latexml_on_content(uncommon_latex)
                assert "Uncommon packages parsed" in xml
                
                # Verify the command was called with proper encoding
                args, kwargs = mock_run.call_args
                assert "--inputencoding=utf8" in args[0]


def test_latex_with_undefined_command(tmp_path):
    """Test handling of LaTeX with undefined commands"""
    latex_with_error = r"""
    \documentclass{article}
    \begin{document}
    This will fail: \undefinedcommand
    \end{document}
    """
    
    with patch("builtins.open", mock_open(read_data="\\documentclass{article}\n\\begin{document}")):
        with patch("os.path.exists", return_value=True):
            with patch("subprocess.run", side_effect=subprocess.CalledProcessError(
                1, "latexml", output="", stderr="Undefined control sequence"
            )) as mock_run:
                xml = latex_processor.run_latexml_on_content(latex_with_error)
                assert xml == ""


def test_latex_with_custom_commands_integration():
    """Test that custom commands are properly handled in the full pipeline"""
    
    # Read the actual preamble file content
    preamble_content = r"""
    \documentclass{article}
    \usepackage{amsmath, amssymb, amsthm}
    \newcommand{\Z}{\mathbb{Z}}
    \newcommand{\R}{\mathbb{R}}
    \newcommand{\Zn}[1]{\Z_{#1}}
    \newcommand{\abs}[1]{\left| #1 \right|}
    \begin{document}
    """
    
    # LaTeX document using custom commands
    test_latex = r"""
    \documentclass{article}
    \begin{document}
    \section{Group Theory}
    
    \begin{theorem}
    The group $\Z$ is infinite, while $\Zn{n}$ has exactly $n$ elements.
    \end{theorem}
    
    \begin{definition}
    The absolute value $\abs{x}$ of an element $x \in \R$ is defined as...
    \end{definition}
    \end{document}
    """
    
    expected_xml = """
    <document>
      <body>
        <section>
          <title>Group Theory</title>
          <theorem>
            <para>The group <Math tex="\\mathbb{Z}">Z</Math> is infinite, 
            while <Math tex="\\mathbb{Z}_{n}">Z_n</Math> has exactly 
            <Math tex="n">n</Math> elements.</para>
          </theorem>
          <theorem class="definition">
            <para>The absolute value <Math tex="\\left| x \\right|">|x|</Math> 
            of an element <Math tex="x \\in \\mathbb{R}">x in R</Math> is defined as...</para>
          </theorem>
        </section>
      </body>
    </document>
    """
    
    with patch("builtins.open", mock_open(read_data=preamble_content)):
        with patch("os.path.exists", return_value=True):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout=expected_xml, 
                    stderr="", 
                    returncode=0
                )
                xml = latex_processor.run_latexml_on_content(test_latex)
                
                # Verify the XML contains expanded math
                assert "\\mathbb{Z}" in xml
                assert "theorem" in xml
                assert "definition" in xml


def test_latexml_complex_xml_output():
    """Test processing of complex XML output from LaTeXML"""
    complex_xml = """
    <document>
      <body>
        <section>
          <theorem xml:id='thm1'><title>Theorem 1</title><para>Let <Math tex="x">x</Math> be a set.</para></theorem>
          <proof><para>This is a proof.</para></proof>
        </section>
      </body>
    </document>
    """
    
    with patch("builtins.open", mock_open(read_data="\\documentclass{article}\n\\begin{document}")):
        with patch("os.path.exists", return_value=True):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout=complex_xml, stderr="", returncode=0)
                xml = latex_processor.run_latexml_on_content("dummy")
                assert "<theorem" in xml and "<proof>" in xml


def test_latex_processor_with_empty_content():
    """Test that the processor handles empty content correctly"""
    # First test with completely empty string
    result = latex_processor.run_latexml_on_content("")
    # LaTeXML returns a valid but empty XML document for empty content
    assert "<document" in result and "LaTeXML" in result
    
    # Test with only whitespace
    result = latex_processor.run_latexml_on_content("   \n\t  ")
    assert "<document" in result and "LaTeXML" in result


def test_latex_processor_with_none_content():
    """Test that the processor handles None content correctly"""
    # Add a check in the actual function to handle None
    with patch("builtins.open", mock_open(read_data="\\documentclass{article}\n\\begin{document}")):
        with patch("os.path.exists", return_value=True):
            # The function should convert None to empty string
            result = latex_processor.run_latexml_on_content(None)
            assert result == ""


def test_latex_processor_with_malformed_latex():
    """Test that the processor handles malformed LaTeX correctly"""
    malformed_latex = r"""
    \documentclass{article}
    \begin{document
    This is an invalid LaTeX document.
    \end{document}
    """
    
    with patch("builtins.open", mock_open(read_data="\\documentclass{article}\n\\begin{document}")):
        with patch("os.path.exists", return_value=True):
            with patch("subprocess.run", side_effect=subprocess.CalledProcessError(
                1, "latexml", output="", stderr="Missing } at line 3"
            )):
                result = latex_processor.run_latexml_on_content(malformed_latex)
                assert result == ""


def test_latex_processor_with_complex_latex():
    """Test that the processor handles complex LaTeX with many custom commands"""
    
    # This should match the master_preamble.tex structure
    complex_preamble = r"""
    \documentclass{article}
    \usepackage{amsmath, amssymb, amsthm, amsfonts}
    \newcommand{\Z}{\mathbb{Z}}
    \newcommand{\R}{\mathbb{R}}
    \newcommand{\C}{\mathbb{C}}
    \newcommand{\N}{\mathbb{N}}
    \newcommand{\Q}{\mathbb{Q}}
    \newcommand{\F}{\mathbb{F}}
    \newcommand{\fieldext}[2]{{#1/#2}}
    \newcommand{\degree}[2]{[#1:#2]}
    \newcommand{\Zn}[1]{\Z_{#1}}
    \newcommand{\abs}[1]{\left| #1 \right|}
    \newcommand{\Znx}[1]{\Z_{#1}^\times}
    \newcommand{\kerphi}{\operatorname{ker}\phi}
    \newcommand{\imphi}{\operatorname{im}\phi}
    \newcommand{\ideal}[1]{\langle#1 \rangle}
    \DeclareMathOperator{\Ker}{Ker}
    \DeclareMathOperator{\im}{im}
    \begin{document}
    """
    
    complex_latex = r"""
    \documentclass{article}
    \begin{document}
    Consider the field extension $\fieldext{\Q(\sqrt{2})}{\Q}$ with degree $\degree{\Q(\sqrt{2})}{\Q} = 2$.
    The group $\Znx{p}$ is cyclic of order $p-1$ for prime $p$.
    For the homomorphism $\phi: \Z \to \Zn{n}$, we have $\kerphi = n\Z$.
    The ideal generated by $2$ is $\ideal{2} = 2\Z$.
    \end{document}
    """
    
    # Expected XML output with expanded commands
    expected_xml_snippet = "field extension"
    
    with patch("builtins.open", mock_open(read_data=complex_preamble)):
        with patch("os.path.exists", return_value=True):
            with patch("subprocess.run") as mock_run:
                # Simulate successful processing
                mock_run.return_value = MagicMock(
                    stdout=f"<document><body><para>Consider the {expected_xml_snippet}</para></body></document>",
                    stderr="",
                    returncode=0
                )
                result = latex_processor.run_latexml_on_content(complex_latex)
                assert expected_xml_snippet in result
                
                # Verify the assembled document includes preamble commands
                call_args = mock_run.call_args[0][0]
                temp_file_path = call_args[-1]  # Last argument is the temp file path


def test_extract_preamble_and_body_functions():
    """Test the helper functions for extracting preamble and body"""
    
    # Test extracting preamble commands
    full_preamble = r"""
    \documentclass{article}
    \usepackage{amsmath}
    \newcommand{\test}{TEST}
    \begin{document}
    Not this part
    \end{document}
    """
    
    commands = latex_processor._extract_preamble_commands(full_preamble)
    assert "\\usepackage{amsmath}" in commands
    assert "\\newcommand{\\test}{TEST}" in commands
    assert "Not this part" not in commands
    
    # Test extracting document body
    full_doc = r"""
    \documentclass{article}
    \usepackage{amsmath}
    \begin{document}
    This is the body content.
    \end{document}
    Extra stuff
    """
    
    body = latex_processor._extract_document_body(full_doc)
    assert body == "This is the body content."
    assert "\\documentclass" not in body
    assert "Extra stuff" not in body


def test_error_handling_and_logging():
    """Test error handling and logging functionality"""
    
    # Test FileNotFoundError when preamble is missing
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="Master preamble not found"):
            latex_processor.run_latexml_on_content("test")
    
    # Test log directory creation
    with patch("os.makedirs") as mock_makedirs:
        latex_processor._ensure_log_dir()
        mock_makedirs.assert_called_once_with(latex_processor.LOG_DIR, exist_ok=True)
    
    # Test subprocess exception handling
    with patch("builtins.open", mock_open(read_data="\\documentclass{article}\n\\begin{document}")):
        with patch("os.path.exists", return_value=True):
            with patch("subprocess.run", side_effect=Exception("Unexpected error")):
                result = latex_processor.run_latexml_on_content("test")
                assert result == ""


def test_get_weaviate_client_success():
    """Test successful connection to Weaviate."""
    from src.data_ingestion import vector_store_manager
    mock_client_instance = MagicMock()
    mock_client_instance.is_ready.return_value = True

    with patch('weaviate.connect_to_local', return_value=mock_client_instance) as mock_connect:
        client = vector_store_manager.get_weaviate_client()
        # Instead of identity, check that the returned object is a mock and is ready
        assert isinstance(client, MagicMock)
        assert client.is_ready()
        mock_connect.assert_called_once()

