import pytest
import os
import tempfile
import re
from src.data_ingestion.latex_parser import parse_latex_file
from src import config

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    test_dir = tempfile.mkdtemp()
    yield test_dir
    # Cleanup after tests
    for root, dirs, files in os.walk(test_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(test_dir)

@pytest.fixture
def create_temp_tex_file(temp_dir):
    """Create a temporary .tex file with given content."""
    def _create_file(content, filename="test.tex"):
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path
    return _create_file

def test_parse_simple_latex(create_temp_tex_file):
    """Test parsing a simple LaTeX document with basic elements."""
    content = """
\\documentclass{article}
% This is a comment.
\\title{Test Title}
\\author{An Author}
\\date{\\today}
\\begin{document}
\\maketitle
Hello World.
This is a test sentence.
An inline formula $E=mc^2$.
\\section{First Section}
Some text in section 1.
Another formula: \\[ \\alpha + \\beta = \\gamma \\]
\\subsection{A Subsection}
More text.
\\end{document}
    """
    file_path = create_temp_tex_file(content)
    parsed_text = parse_latex_file(file_path)

    assert "Hello World." in parsed_text
    assert "This is a test sentence." in parsed_text
    assert "$E=mc^2$" in parsed_text, "Inline math should be preserved"
    assert "% This is a comment." not in parsed_text, "Comments should be removed"
    
    assert "FIRST SECTION" in parsed_text, "Section title text should be present"
    assert "\\section{First Section}" not in parsed_text, "Raw section command should be converted"
    assert "§" in parsed_text, "Default section marker '§' should be present"

    assert "\\[ \\alpha + \\beta = \\gamma \\]" in parsed_text, "Display math should be preserved"
    
    assert "Test Title" not in parsed_text, "Title content should be removed"
    assert "An Author" not in parsed_text, "Author content should be removed"
    
    assert "\\documentclass{article}" not in parsed_text

def test_empty_file(create_temp_tex_file):
    """Test parsing an empty LaTeX file."""
    file_path = create_temp_tex_file("")
    parsed_text = parse_latex_file(file_path)
    assert parsed_text == ""

def test_file_not_found():
    """Test handling of non-existent file."""
    parsed_text = parse_latex_file("non_existent_file.tex")
    assert parsed_text == ""

def test_math_environments(create_temp_tex_file):
    """Test parsing various LaTeX math environments."""
    content = """
\\begin{document}
Inline: $a^2+b^2=c^2$.
Display: \\[ x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a} \\]
Equation:
\\begin{equation}
\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\epsilon_0}
\\end{equation}
Align:
\\begin{align}
f(x) &= (x+a)(x+b) \\\\
     &= x^2 + (a+b)x + ab
\\end{align}
\\end{document}
    """
    file_path = create_temp_tex_file(content)
    parsed_text = parse_latex_file(file_path)
    
    assert "$a^2+b^2=c^2$" in parsed_text
    assert "\\[ x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a} \\]" in parsed_text
    assert "\\begin{equation}\n\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\epsilon_0}\n\\end{equation}" in parsed_text
    # Adjust assertion for align to be more flexible with internal whitespace
    assert re.search(
        r"\\begin\{align\}\s*f\(x\) &= \(x\+a\)\(x\+b\)\s*\\\\\s*&= x\^2 \+ \(a\+b\)x \+ ab\s*\\end\{align\}",
        parsed_text
    ), f"Align environment not found or formatted unexpectedly in:\n{parsed_text}"

def test_structural_commands(create_temp_tex_file):
    """Test parsing LaTeX structural commands."""
    content = """
\\documentclass{article}
\\begin{document}
\\section{Main Section}
Text here.
\\subsection{Subsection A}
More text.
\\subsubsection{Subsubsection B}
Details.
\\paragraph{A Paragraph Title}
Paragraph content.
\\end{document}
    """
    file_path = create_temp_tex_file(content)
    parsed_text = parse_latex_file(file_path)

    assert "MAIN SECTION" in parsed_text
    assert "\\section{Main Section}" not in parsed_text
    assert "§ MAIN SECTION" in parsed_text

    assert "Subsection A" in parsed_text
    assert "\\subsection{Subsection A}" not in parsed_text
    assert "§.§ Subsection A" in parsed_text

    assert "Subsubsection B" in parsed_text
    assert "\\subsubsection{Subsubsection B}" not in parsed_text
    assert "§.§.§ Subsubsection B" in parsed_text

    assert "A Paragraph Title" in parsed_text
    assert "Paragraph content." in parsed_text
    assert "\\paragraph{A Paragraph Title}" not in parsed_text
    assert re.search(
        r"A Paragraph Title\s*\n\s*Paragraph content.",
        parsed_text,
        re.IGNORECASE
    ), "Paragraph title and content format not as expected." 