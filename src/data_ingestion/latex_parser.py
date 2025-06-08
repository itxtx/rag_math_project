# src/data_ingestion/latex_parser.py
import os
import re
import uuid
import pickle
import networkx as nx
from pylatexenc.latex2text import LatexNodes2Text
from sentence_transformers import SentenceTransformer

# This is our new, more robust processor
from . import latex_processor

class LatexToGraphParser:
    """
    Parses a pre-processed LaTeX document to extract structured nodes 
    (theorems, definitions, etc.) and their relationships, building a knowledge graph.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        # We'll build the graph as we parse documents
        self.graph = nx.DiGraph() 

    def extract_structured_nodes(self, latex_content: str, doc_id: str, source: str = None):
        """
        Extracts environments like theorem, definition, etc., as nodes for the graph.
        
        Args:
            latex_content: The LaTeX content to process
            doc_id: Unique identifier for the document
            source: Optional source path of the document
        """
        # First, pre-process the entire document body with LaTeXML
        # This gives us clean text with standard math notation.
        clean_content = latex_processor.process_latex_document(latex_content)

        if not clean_content:
            print(f"WARNING: LaTeXML returned no content for doc_id: {doc_id}")
            return

        # Now, use pylatexenc to find the structure (environments)
        # We run it on the *original* content to find the structure, but use the
        # *clean* content for the text. This is a hybrid approach.
        converter = LatexNodes2Text()
        nodes = converter.latex_to_text(latex_content)

        # Process each environment
        for env_name in ['theorem', 'lemma', 'definition', 'proof', 'corollary']:
            # Find all instances of this environment
            pattern = rf'\\begin{{{env_name}}}(.*?)\\end{{{env_name}}}'
            matches = re.finditer(pattern, latex_content, re.DOTALL)
            
            for match in matches:
                env_content_raw = match.group(0)
                # Get the clean text from LaTeXML for this specific snippet
                env_content_clean = latex_processor.process_latex_document(env_content_raw)
                
                if not env_content_clean:
                    continue

                # Find label for ID, and refs for edges
                label = self._find_label(env_content_raw) or f"{env_name}-{uuid.uuid4().hex[:8]}"
                refs = self._find_refs(env_content_raw)

                # Generate the initial text embedding
                embedding = self.embedding_model.encode(env_content_clean, convert_to_tensor=False)

                # Add the node to the graph
                self.graph.add_node(
                    label,
                    node_type=env_name,
                    doc_id=doc_id,
                    source=source,  # Add source to node attributes
                    text=env_content_clean,
                    embedding=embedding
                )
                
                # Add edges for each reference found
                for ref_label in refs:
                    # We add the edge now, even if the 'ref_label' node doesn't exist yet.
                    # It will be created when its own environment is parsed.
                    self.graph.add_edge(label, ref_label, edge_type='references')

    def _find_label(self, content: str) -> str:
        """Find the label in the content."""
        label_match = re.search(r'\\label{([^}]+)}', content)
        return label_match.group(1) if label_match else None

    def _find_refs(self, content: str) -> list:
        """Find all references in the content."""
        refs = []
        for ref_type in ['ref', 'eqref']:
            ref_matches = re.finditer(rf'\\{ref_type}{{{{([^}}]+)}}}}', content)
            
            refs.extend(match.group(1) for match in ref_matches)
        return refs

    def save_graph_and_embeddings(self, graph_path, embeddings_path):
        """Saves the final graph and initial embeddings."""
        print(f"Saving knowledge graph to {graph_path}")
        nx.write_graphml(self.graph, graph_path)

        initial_embeddings = {node: data['embedding'] for node, data in self.graph.nodes(data=True)}
        print(f"Saving initial text embeddings to {embeddings_path}")
        with open(embeddings_path, 'wb') as f:
            pickle.dump(initial_embeddings, f)
