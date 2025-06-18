import pytest
import networkx as nx
from unittest.mock import patch
from src.data_ingestion.latex_parser import LatexToGraphParser

@pytest.fixture
def parser():
    """Fixture to create a LatexToGraphParser instance."""
    return LatexToGraphParser()

def test_initialization(parser):
    """Test that the parser is initialized correctly."""
    assert parser.graph is not None
    assert isinstance(parser.graph, nx.DiGraph)
    assert parser.embedding_model is not None

@patch('src.data_ingestion.latex_parser.latex_processor.run_latexml_on_content')
def test_graph_creation_from_xml(mock_run_latexml, parser):
    """Test that a graph is created from a mocked LaTeXML output."""
    # Wrap the theorem in a more realistic structure that the parser expects
    mock_run_latexml.return_value = """
    <document>
    <body>
    <section>
    <theorem xml:id="Thm1" class="ltx_theorem_theorem">
    <title><tag>Theorem 1.1</tag> <title>Title of Theorem</title></title>
    <para>This is the content of the theorem.</para>
    </theorem>
    </section>
    </body>
    </document>
    """
    latex_string = r"\section{Section 1}\begin{theorem}[Title]Content\end{theorem}"
    parser.extract_structured_nodes(latex_string, "Test Document", "source.tex")
    
    assert len(parser.graph.nodes) >= 1, "The graph should have at least one node."
    node_id = list(parser.graph.nodes)[0]
    node_data = parser.graph.nodes[node_id]
    
    assert node_data['node_type'] == 'theorem'
    assert "Title of Theorem" in node_data['title']
    assert "content of the theorem" in node_data['text']
    assert node_data['doc_id'] == "Test Document"

@patch('src.data_ingestion.latex_parser.latex_processor.run_latexml_on_content')
def test_no_structured_content(mock_run_latexml, parser):
    """Test parsing LaTeX with no structured environments."""
    mock_run_latexml.return_value = "<para>Just some plain text without any sections.</para>"
    latex_string = "Just some plain text without any sections."
    parser.extract_structured_nodes(latex_string, "Plain Document", "plain.tex")
    assert len(parser.graph.nodes) == 0

def test_get_clean_text(parser):
    """This test is a placeholder for more detailed testing."""
    pass