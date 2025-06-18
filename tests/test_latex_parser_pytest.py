import pytest
import networkx as nx
from src.data_ingestion.latex_parser import LatexToGraphParser

@pytest.fixture
def parser():
    """Fixture to create a LatexToGraphParser instance."""
    return LatexToGraphParser()

def test_initialization(parser):
    """Test that the parser is initialized correctly."""
    assert parser.section_pattern.pattern == r'\\(section|subsection|subsubsection|paragraph|subparagraph)\*?\{([^}]+)\}'
    assert parser.item_pattern.pattern == r'\\item'

def test_graph_creation_with_sections(parser):
    """Test that a graph is created correctly from a simple LaTeX string with sections."""
    latex_string = r"""
    \section{Section 1}
    Some text here.
    \subsection{Subsection A}
    More text.
    """
    parser.extract_structured_nodes(latex_string, "Test Document")
    graph = parser.graph
    assert isinstance(graph, nx.DiGraph)
    assert len(graph.nodes) > 0
    # Check that a node with the title "Section 1" exists
    assert "Section 1" in [data.get('title') for _, data in graph.nodes(data=True)]

def test_graph_creation_with_items(parser):
    """Test that a graph is created correctly from a LaTeX string with items."""
    latex_string = r"""
    \section{List Section}
    \begin{itemize}
        \item First item.
        \item Second item.
    \end{itemize}
    """
    parser.extract_structured_nodes(latex_string, "Test Document")
    graph = parser.graph
    nodes_data = [data for _, data in graph.nodes(data=True)]
    # Check for nodes that are items (have no title but content)
    item_nodes = [d for d in nodes_data if d.get('title') is None and 'First item' in d.get('content', '')]
    assert len(item_nodes) > 0

def test_node_attributes(parser):
    """Test that nodes in the graph have the correct attributes."""
    latex_string = r"\section{Main Section}"
    parser.extract_structured_nodes(latex_string, "Test Document")
    graph = parser.graph
    # Get the data of the first node (assuming one node is created)
    node_data = list(graph.nodes(data=True))[1][1] # Index 0 is the document node
    assert 'title' in node_data
    assert 'content' in node_data
    assert node_data['title'] == "Main Section"

def test_edge_creation(parser):
    """Test that edges are created correctly between sections and subsections."""
    latex_string = r"""
    \section{Parent}
    \subsection{Child}
    """
    parser.extract_structured_nodes(latex_string, "Test Document")
    graph = parser.graph
    # Find nodes by title
    parent_node = [n for n, d in graph.nodes(data=True) if d.get('title') == "Parent"][0]
    child_node = [n for n, d in graph.nodes(data=True) if d.get('title') == "Child"][0]
    # Check that an edge exists from parent to child
    assert graph.has_edge(parent_node, child_node)

def test_document_node(parser):
    """Test that a document-level node is created."""
    latex_string = r"\section{A Section}"
    parser.extract_structured_nodes(latex_string, "My Test Doc")
    graph = parser.graph
    doc_node = [n for n, d in graph.nodes(data=True) if d.get('title') == "My Test Doc"][0]
    section_node = [n for n, d in graph.nodes(data=True) if d.get('title') == "A Section"][0]
    # The document node should be a predecessor to the section node
    assert doc_node in graph.predecessors(section_node)

def test_no_sections(parser):
    """Test parsing a LaTeX string with no sections."""
    latex_string = "Just some plain text without any sections."
    parser.extract_structured_nodes(latex_string, "Plain Document")
    graph = parser.graph
    # Expect a single document node with the content
    assert len(graph.nodes) == 1
    node_data = next(iter(graph.nodes(data=True)))[1]
    assert node_data['title'] == "Plain Document"
    assert "Just some plain text" in node_data['content']

def test_nested_sections(parser):
    """Test deeply nested section hierarchy."""
    latex_string = r"""
    \section{Level 1}
    \subsection{Level 2}
    \subsubsection{Level 3}
    """
    parser.extract_structured_nodes(latex_string, "Nested Document")
    graph = parser.graph
    level1 = [n for n, d in graph.nodes(data=True) if d.get('title') == "Level 1"][0]
    level2 = [n for n, d in graph.nodes(data=True) if d.get('title') == "Level 2"][0]
    level3 = [n for n, d in graph.nodes(data=True) if d.get('title') == "Level 3"][0]
    assert graph.has_edge(level1, level2)
    assert graph.has_edge(level2, level3)

def test_starred_sections(parser):
    """Test that starred versions of sections (e.g., \section*{...}) are parsed correctly."""
    latex_string = r"\section*{Starred Section}"
    parser.extract_structured_nodes(latex_string, "Starred Test")
    graph = parser.graph
    # Check that the title "Starred Section" is present in one of the nodes
    assert "Starred Section" in [data.get('title') for _, data in graph.nodes(data=True)]

def test_empty_content(parser):
    """Test a LaTeX string where sections have no content between them."""
    latex_string = r"""
    \section{Section A}
    \subsection{Subsection B}
    """
    parser.extract_structured_nodes(latex_string, "Empty Content Test")
    graph = parser.graph
    section_a_data = [d for _, d in graph.nodes(data=True) if d.get('title') == "Section A"][0]
    # The content of Section A should be empty or just whitespace
    assert section_a_data['content'].strip() == ""

@pytest.mark.complex
def test_complex_document_structure(parser):
    """
    Test a more complex document with mixed sections, lists, and content.
    This test ensures that the hierarchical relationships and content assignments are correct.
    """
    latex_string = r"""
    \section{Introduction}
    This is the introduction.
    \begin{itemize}
        \item Point one.
        \item Point two.
    \end{itemize}
    \section{Methodology}
    This section describes the methodology.
    \subsection{Subsection A}
    Details of subsection A.
    """
    parser.extract_structured_nodes(latex_string, "Complex Doc")
    graph = parser.graph

    # Verify node counts
    assert len(graph.nodes) == 6  # Doc, Intro, Item1, Item2, Method, SubA

    # Get nodes by title
    intro = [n for n, d in graph.nodes(data=True) if d.get('title') == "Introduction"][0]
    method = [n for n, d in graph.nodes(data=True) if d.get('title') == "Methodology"][0]
    sub_a = [n for n, d in graph.nodes(data=True) if d.get('title') == "Subsection A"][0]
    item_nodes = [n for n, d in graph.nodes(data=True) if d.get('title') is None and "Point" in d.get('content', '')]

    # Check edges
    assert graph.has_edge(intro, item_nodes[0])
    assert graph.has_edge(intro, item_nodes[1])
    assert graph.has_edge(method, sub_a)

    # Check content parsing
    intro_data = graph.nodes[intro]
    assert "This is the introduction" in intro_data['content']
    
    # Check that the title is correctly parsed for a specific node
    assert any("Subsection A" in data.get('title', '') for _, data in graph.nodes(data=True) if data.get('title') == "Subsection A")

@pytest.mark.edge_case
def test_latex_with_special_characters_in_title(parser):
    """
    Test that section titles with LaTeX special characters are handled correctly.
    This ensures the regex and parsing logic are robust.
    """
    latex_string = r"\section{A title with $maths$ and \& symbols}"
    parser.extract_structured_nodes(latex_string, "Special Chars Test")
    graph = parser.graph

    # Check if the title is parsed correctly, including the special characters
    parsed_titles = [d.get('title') for _, d in graph.nodes(data=True) if d.get('title')]
    expected_title = "A title with $maths$ and & symbols"
    assert expected_title in parsed_titles

@pytest.mark.xfail(reason="Content after the last section is currently ignored.")
def test_content_after_last_section(parser):
    """
    Test for content that appears after the very last section or item.
    This is an expected failure as the current implementation might not capture trailing content.
    """
    latex_string = r"""
    \section{Final Section}
    Content of the final section.
    Trailing content that should be captured.
    """
    parser.extract_structured_nodes(latex_string, "Trailing Content Test")
    graph = parser.graph
    final_section_data = [d for _, d in graph.nodes(data=True) if d.get('title') == "Final Section"][0]
    assert "Trailing content" in final_section_data['content']