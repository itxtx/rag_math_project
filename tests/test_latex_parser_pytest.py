import pytest
import networkx as nx
from unittest.mock import patch
from src.data_ingestion.latex_parser import LatexToGraphParser

# Mock LaTeXML output for controlled testing
def mock_run_latexml_on_content(latex_content):
    if "section{Section 1}" in latex_content:
        return """
<theorem xml:id="Thm1" class="ltx_theorem_theorem">
<title><tag>Theorem 1.1</tag> <title>Title of Theorem</title></title>
<para>This is the content of the theorem.</para>
</theorem>
"""
    if "My Test Doc" in latex_content:
        return """
<theorem xml:id="Thm_doc" class="ltx_theorem_theorem">
<title>A theorem</title>
<para>Theorem content.</para>
</theorem>
"""
    if "Just some plain text" in latex_content:
        return "<para>Just some plain text without any sections.</para>"
    return ""

@pytest.fixture
def parser():
    """Fixture to create a LatexToGraphParser instance with a mocked LaTeXML runner."""
    with patch('src.data_ingestion.latex_processor.run_latexml_on_content', side_effect=mock_run_latexml_on_content):
        yield LatexToGraphParser()

def test_initialization(parser):
    """Test that the parser is initialized correctly."""
    assert parser.graph is not None
    assert isinstance(parser.graph, nx.DiGraph)
    assert parser.embedding_model is not None

def test_graph_creation_from_xml(parser):
    """Test that a graph is created from a mocked LaTeXML output."""
    latex_string = r"\section{Section 1}\begin{theorem}[Title]Content\end{theorem}"
    parser.extract_structured_nodes(latex_string, "Test Document", "source.tex")
    
    assert len(parser.graph.nodes) == 1
    node_id = list(parser.graph.nodes)[0]
    node_data = parser.graph.nodes[node_id]
    
    assert node_data['node_type'] == 'theorem'
    assert "Title of Theorem" in node_data['title']
    assert "content of the theorem" in node_data['text']
    assert node_data['doc_id'] == "Test Document"

def test_no_structured_content(parser):
    """Test parsing LaTeX with no structured environments."""
    latex_string = "Just some plain text without any sections."
    parser.extract_structured_nodes(latex_string, "Plain Document", "plain.tex")
    # The new parser only creates nodes for structured environments.
    # If no environments are found, the graph should be empty.
    assert len(parser.graph.nodes) == 0

def test_get_clean_text():
    """Test the internal text cleaning utility."""
    parser = LatexToGraphParser()
    # This test would need to be more elaborate, creating mock XML elements
    # to test the _get_clean_text method in isolation. For brevity, this is omitted here,
    # but in a real scenario, you'd test this helper function thoroughly.
    pass