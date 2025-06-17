import pytest
import os
import tempfile
import re
from src.data_ingestion.latex_parser import parse_latex_file
from src import config
from src.data_ingestion.latex_to_graph_parser import LatexToGraphParser

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
    parser = LatexToGraphParser()
    parser.extract_structured_nodes(content, "test_doc", file_path)
    graph = parser.graph

    assert graph.number_of_nodes() > 0

    # Example of how to check for content in the graph
    found_hello = any("Hello World" in data.get('text', '') for _, data in graph.nodes(data=True))
    assert found_hello

    found_section = any("First Section" in data.get('title', '') for _, data in graph.nodes(data=True))
    assert found_section

    assert any("$E=mc^2$" in data.get('text', '') for _, data in graph.nodes(data=True))
    assert all("% This is a comment." not in data.get('text', '') for _, data in graph.nodes(data=True))
    assert any("FIRST SECTION" in data.get('title', '') for _, data in graph.nodes(data=True))
    assert all("\\section{First Section}" not in data.get('text', '') for _, data in graph.nodes(data=True))
    assert any("§" in data.get('title', '') for _, data in graph.nodes(data=True))
    assert any("\\[ \\alpha + \\beta = \\gamma \\]" in data.get('text', '') for _, data in graph.nodes(data=True))
    assert all("Test Title" not in data.get('title', '') for _, data in graph.nodes(data=True))
    assert all("An Author" not in data.get('title', '') for _, data in graph.nodes(data=True))
    assert all("\\documentclass{article}" not in data.get('text', '') for _, data in graph.nodes(data=True))

    assert any("Subsection A" in data.get('title', '') for _, data in graph.nodes(data=True))
    assert all("\\subsection{Subsection A}" not in data.get('text', '') for _, data in graph.nodes(data=True))
    assert any("§.§ Subsection A" in data.get('title', '') for _, data in graph.nodes(data=True))
    assert any("Subsubsection B" in data.get('title', '') for _, data in graph.nodes(data=True))
    assert all("\\subsubsection{Subsubsection B}" not in data.get('text', '') for _, data in graph.nodes(data=True))
    assert any("§.§.§ Subsubsection B" in data.get('title', '') for _, data in graph.nodes(data=True))
    assert any("A Paragraph Title" in data.get('title', '') for _, data in graph.nodes(data=True))
    assert any("Paragraph content." in data.get('text', '') for _, data in graph.nodes(data=True))
    assert all("\\paragraph{A Paragraph Title}" not in data.get('text', '') for _, data in graph.nodes(data=True))
    assert any(re.search(r"A Paragraph Title\s*\n\s*Paragraph content.", data.get('text', ''), re.IGNORECASE) for _, data in graph.nodes(data=True))
    assert all("\\documentclass{article}" not in data.get('text', '') for _, data in graph.nodes(data=True))

def test_empty_file(create_temp_tex_file):
    """Test parsing an empty LaTeX file."""
    file_path = create_temp_tex_file("")
    parser = LatexToGraphParser()
    parser.extract_structured_nodes("", "test_doc", file_path)
    graph = parser.graph
    assert graph.number_of_nodes() == 0

def test_file_not_found():
    """Test handling of non-existent file."""
    with pytest.raises(FileNotFoundError):
        parse_latex_file("non_existent_file.tex")

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
    parser = LatexToGraphParser()
    parser.extract_structured_nodes(content, "test_doc", file_path)
    graph = parser.graph

    assert graph.number_of_nodes() > 0

    # Example of how to check for math content in the graph
    found_inline = any("$a^2+b^2=c^2$" in data.get('text', '') for _, data in graph.nodes(data=True))
    assert found_inline

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
    parser = LatexToGraphParser()
    parser.extract_structured_nodes(content, "test_doc", file_path)
    graph = parser.graph

    assert graph.number_of_nodes() > 0

    # Example of how to check for structural content in the graph
    found_main_section = any("Main Section" in data.get('title', '') for _, data in graph.nodes(data=True))
    assert found_main_section

    assert "Subsection A" in data.get('title', '') for _, data in graph.nodes(data=True)
    assert "\\subsection{Subsection A}" not in data.get('text', '') for _, data in graph.nodes(data=True)
    assert "§.§ Subsection A" in data.get('title', '') for _, data in graph.nodes(data=True)

    assert "Subsubsection B" in data.get('title', '') for _, data in graph.nodes(data=True)
    assert "\\subsubsection{Subsubsection B}" not in data.get('text', '') for _, data in graph.nodes(data=True)
    assert "§.§.§ Subsubsection B" in data.get('title', '') for _, data in graph.nodes(data=True)

    assert "A Paragraph Title" in data.get('title', '') for _, data in graph.nodes(data=True)
    assert "Paragraph content." in data.get('text', '') for _, data in graph.nodes(data=True)
    assert "\\paragraph{A Paragraph Title}" not in data.get('text', '') for _, data in graph.nodes(data=True)
    assert re.search(
        r"A Paragraph Title\s*\n\s*Paragraph content.",
        data.get('text', ''),
        re.IGNORECASE
    ) for _, data in graph.nodes(data=True)
    
    assert "\\documentclass{article}" not in data.get('text', '') for _, data in graph.nodes(data=True) 